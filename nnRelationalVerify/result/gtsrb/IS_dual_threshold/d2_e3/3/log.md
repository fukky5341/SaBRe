## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 3600 seconds
Split limit: 100
Threshold: 37.4874905906


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341)
1: (-23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751)
2: (-18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7670898, 50.7670937)
3: (-19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7754364, 53.7754440)
4: (-23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012)
5: (-21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0920868, 56.0921021)
6: (-42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501)
7: (-30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0413132, 64.0413132)
8: (-29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192)
9: (-24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9636116, 54.9636154)
10: (-45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318)
11: (-48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746)
12: (-52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4354858, 69.4354935)
13: (-35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266)
14: (-78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879)
15: (-30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065)
16: (-46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2572174, 77.2572327)
17: (-77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000)
18: (-45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464)
19: (-34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724)
20: (-30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395)
21: (-42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793)
22: (-43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864)
23: (-34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452)
24: (-36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056)
25: (-35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590)
26: (-53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457)
27: (-36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078)
28: (-33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381)
29: (-45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077)
30: (-42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827)
31: (-42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359)
32: (-38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531)
33: (-48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046)
34: (-47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2930145, 68.2930222)
35: (-41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7198257, 67.7198334)
36: (-42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2822266, 68.2822189)
37: (-66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7244415, 86.7244415)
38: (-52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0830994, 82.0831070)
39: (-60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865)
40: (-53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266)
41: (-39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605)
42: (-32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.81 + 77.25 = 80.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -37.5438063, upper bound: 37.5438063

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5358797, upper bound: 37.5150455
time: 56.55 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5358797, upper bound: 37.5358796
time: 51.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 108.09 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 108.09
Output dim: 8, lower bound: -37.5358797, upper bound: 37.5150455
IS_A2, status: Status.UNKNOWN, split count: 1, time: 108.09
Output dim: 8, lower bound: -37.5358797, upper bound: 37.5358796

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -43.0271416, 35.1652374, -43.1028595, 35.2126312, -78.2397766, 78.2680969
1: -23.2989120, 31.9865475, -23.3465614, 32.0159454, -55.3148575, 55.3331070
2: -18.7985229, 31.8990250, -18.8387070, 31.9232903, -50.7003784, 50.7185020
3: -19.0018692, 35.1514435, -19.0250435, 35.1672440, -53.7320099, 53.7434845
4: -23.4423103, 35.9524918, -23.5046120, 35.9962120, -59.4385223, 59.4571037
5: -21.1640930, 35.4586143, -21.1939983, 35.4933891, -56.0232086, 56.0271568
6: -42.1417236, 26.0500813, -42.1863403, 26.0977211, -68.2394409, 68.2364197
7: -30.3642521, 34.1984024, -30.4128380, 34.2341309, -63.9441528, 63.9701080
8: -28.9593830, 40.0518570, -29.0183163, 40.0991783, -69.0585632, 69.0701752
9: -24.3802738, 31.6214046, -24.4144363, 31.6499748, -54.8770599, 54.9079590
10: -45.8472519, 31.2726593, -45.8814087, 31.3038788, -77.1511307, 77.1540680
11: -48.8224106, 18.1337891, -48.8816681, 18.2197895, -67.0421982, 67.0154572
12: -52.8201180, 18.1903954, -52.8530922, 18.2210331, -69.3770370, 69.3506851
13: -35.7417297, 38.6799850, -35.7891922, 38.7014542, -74.4431839, 74.4691772
14: -78.4322357, 11.0370770, -78.4726715, 11.0843105, -89.5165482, 89.5097504
15: -30.3528881, 30.1095695, -30.3961182, 30.1348209, -60.4877090, 60.5056877
16: -46.2164917, 30.8870316, -46.2845116, 30.9341240, -77.1401672, 77.1634979
17: -77.8508530, 14.5910473, -77.9211349, 14.7115231, -92.5623779, 92.5121841
18: -45.8193359, 21.2626228, -45.8263016, 21.3092499, -67.1285858, 67.0889282
19: -34.4492950, 10.9337301, -34.4950409, 10.9970570, -45.4463501, 45.4287720
20: -30.5701160, 14.2747974, -30.6001167, 14.3133965, -44.8835144, 44.8749161
21: -42.6164818, 14.8743086, -42.6784286, 14.9630108, -57.5794907, 57.5527382
22: -43.2259216, 17.5408020, -43.3012657, 17.6400986, -60.8660202, 60.8420677
23: -34.3449745, 15.1079302, -34.3971291, 15.1877880, -49.5327606, 49.5050583
24: -36.3411560, 14.8657827, -36.3916779, 14.9165039, -51.2576599, 51.2574615
25: -35.4938622, 17.2701359, -35.5531502, 17.3575554, -52.8514175, 52.8232880
26: -53.5420761, 20.2093697, -53.5872116, 20.2753372, -73.8174133, 73.7965851
27: -36.2236710, 18.9240494, -36.2385864, 18.9616547, -55.1853256, 55.1626358
28: -33.2716942, 18.9635887, -33.3143387, 19.0323792, -52.3040733, 52.2779274
29: -44.8522835, 16.7659645, -44.9446335, 16.8928089, -61.7450943, 61.7105980
30: -42.7712517, 19.9888725, -42.8357239, 20.0743542, -62.8456039, 62.8245964
31: -42.2673569, 15.3061113, -42.3103752, 15.3618965, -57.6292534, 57.6164856
32: -38.4951820, 23.1559181, -38.5328369, 23.1876984, -61.6828804, 61.6887550
33: -48.8172684, 35.9686928, -48.8745384, 35.9802933, -84.7975616, 84.8432312
34: -47.1526337, 21.1096153, -47.1851387, 21.1311264, -68.2470245, 68.2581253
35: -41.6700745, 26.4139957, -41.7095413, 26.4360237, -67.6506805, 67.6637955
36: -42.4433975, 26.6229038, -42.4777031, 26.6414070, -68.2229919, 68.2471771
37: -66.8369675, 22.2697964, -66.8885956, 22.3046589, -86.6076202, 86.6377182
38: -52.5261688, 31.1963654, -52.5858269, 31.2471237, -81.9454269, 81.9667282
39: -60.2385330, 35.4226379, -60.3112717, 35.4476624, -95.6861954, 95.7339096
40: -53.4567490, 28.2990799, -53.5675430, 28.3700943, -81.8268433, 81.8666229
41: -39.0631561, 27.1278229, -39.1124802, 27.1645355, -66.2276917, 66.2403030
42: -32.5265121, 21.9982643, -32.5523796, 22.0131569, -54.5396690, 54.5506439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 615

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5279812, upper bound: 37.4915721
time: 158.25 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5279812, upper bound: 37.5059792
time: 62.76 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -43.1133041, 35.2487221, -43.1144791, 35.2513123, -78.3646164, 78.3632050
1: -23.3516788, 32.0366440, -23.3521214, 32.0382652, -55.3899460, 55.3887634
2: -18.8431225, 31.9390640, -18.8435783, 31.9404602, -50.7657242, 50.7624435
3: -19.0243187, 35.1731720, -19.0283985, 35.1743660, -53.7787476, 53.7680435
4: -23.5092068, 36.0238266, -23.5103779, 36.0274658, -59.5366745, 59.5342026
5: -21.1975937, 35.5144501, -21.1982746, 35.5162125, -56.0940704, 56.0836334
6: -42.1921654, 26.1272545, -42.1929169, 26.1302414, -68.3224030, 68.3201752
7: -30.4184570, 34.2577400, -30.4190941, 34.2595901, -64.0448608, 64.0317078
8: -29.0218735, 40.1287498, -29.0224819, 40.1319351, -69.1538086, 69.1512299
9: -24.4184914, 31.6681976, -24.4192810, 31.6693821, -54.9778481, 54.9497375
10: -45.8835220, 31.3122520, -45.8888550, 31.3131981, -77.1967163, 77.2011108
11: -48.9130669, 18.2230721, -48.9174423, 18.2240181, -67.1370850, 67.1405182
12: -52.8654099, 18.2242641, -52.8676949, 18.2251053, -69.4203796, 69.4475708
13: -35.7852173, 38.7079964, -35.7951164, 38.7093544, -74.4945679, 74.5031128
14: -78.4911957, 11.0877075, -78.4932632, 11.0883942, -89.5795898, 89.5809708
15: -30.4129066, 30.1410904, -30.4150906, 30.1420708, -60.5549774, 60.5561829
16: -46.2934036, 30.9633865, -46.2944221, 30.9661560, -77.2541351, 77.2496567
17: -77.9599304, 14.7179489, -77.9624634, 14.7187691, -92.6786957, 92.6804123
18: -45.8317642, 21.3127747, -45.8335876, 21.3248711, -67.1566315, 67.1463623
19: -34.5215263, 10.9998360, -34.5238266, 11.0001860, -45.5217133, 45.5236626
20: -30.6152592, 14.3160658, -30.6167698, 14.3163891, -44.9316483, 44.9328346
21: -42.7139359, 14.9655657, -42.7167778, 14.9660015, -57.6799393, 57.6823425
22: -43.3492622, 17.6436062, -43.3525124, 17.6440125, -60.9932747, 60.9961166
23: -34.4312592, 15.1903419, -34.4334221, 15.1911678, -49.6224289, 49.6237640
24: -36.4245377, 14.9186497, -36.4267731, 14.9188538, -51.3433914, 51.3454208
25: -35.5953979, 17.3621521, -35.5978165, 17.3628273, -52.9582253, 52.9599686
26: -53.6105766, 20.2785969, -53.6127243, 20.2791882, -73.8897629, 73.8913193
27: -36.2463226, 18.9579182, -36.2475357, 18.9656830, -55.2120056, 55.2054520
28: -33.3412666, 19.0350170, -33.3433990, 19.0357590, -52.3770256, 52.3784180
29: -45.0076141, 16.8942738, -45.0110130, 16.8957710, -61.9033852, 61.9052887
30: -42.8758659, 20.0776043, -42.8783569, 20.0786858, -62.9545517, 62.9559631
31: -42.3346024, 15.3654423, -42.3370171, 15.3657751, -57.7003784, 57.7024612
32: -38.5398254, 23.2062798, -38.5415421, 23.2074280, -61.7472534, 61.7478218
33: -48.8745193, 35.9854050, -48.8833885, 35.9863091, -84.8608246, 84.8687897
34: -47.1886215, 21.1357193, -47.1899529, 21.1369629, -68.2892151, 68.2886810
35: -41.7194977, 26.4403000, -41.7238884, 26.4413528, -67.7036285, 67.7136307
36: -42.4821167, 26.6430359, -42.4836197, 26.6452332, -68.2765274, 68.2643738
37: -66.8995819, 22.3196640, -66.9010849, 22.3258095, -86.7194824, 86.6706161
38: -52.5905800, 31.2642384, -52.5935898, 31.2742100, -82.0736542, 82.0373154
39: -60.3183174, 35.4639130, -60.3209724, 35.4651031, -95.7834167, 95.7848816
40: -53.5763321, 28.4226379, -53.5780106, 28.4278851, -82.0042191, 82.0006485
41: -39.1180992, 27.1867981, -39.1187096, 27.1891251, -66.3072205, 66.3055115
42: -32.5546875, 22.0185223, -32.5587387, 22.0197887, -54.5744781, 54.5772629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 615

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5106837, upper bound: 37.5279811
time: 54.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5106837, upper bound: 37.5279811
time: 60.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 117.99 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 117.99
Output dim: 8, lower bound: -37.5279812, upper bound: 37.4915721
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 117.99
Output dim: 8, lower bound: -37.5279812, upper bound: 37.5059792
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 117.99
Output dim: 8, lower bound: -37.5106837, upper bound: 37.5279811
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 117.99
Output dim: 8, lower bound: -37.5106837, upper bound: 37.5279811

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -43.0020485, 35.0917587, -43.0901337, 35.1751060, -78.1771545, 78.1818924
1: -23.2879124, 31.9523735, -23.3410034, 31.9980030, -55.2859154, 55.2933769
2: -18.7863388, 31.8679771, -18.8325462, 31.9076920, -50.6704559, 50.6787682
3: -18.9752750, 35.1377716, -19.0117702, 35.1602097, -53.6961746, 53.7142639
4: -23.4250221, 35.8884659, -23.4959049, 35.9641380, -59.3852768, 59.3838272
5: -21.1520939, 35.4208145, -21.1878815, 35.4739990, -55.9848785, 55.9723854
6: -42.1281357, 25.9984417, -42.1794014, 26.0719681, -68.2001038, 68.1778412
7: -30.3516312, 34.1688919, -30.4064026, 34.2191620, -63.9060135, 63.9191895
8: -28.9521427, 39.9950409, -29.0146427, 40.0702324, -69.0223770, 69.0096817
9: -24.3613930, 31.6007347, -24.4048920, 31.6390610, -54.8498192, 54.8660278
10: -45.7943153, 31.2555542, -45.8548737, 31.2949677, -77.0892792, 77.1104279
11: -48.7463989, 18.1251907, -48.8431168, 18.2154045, -66.9618073, 66.9683075
12: -52.7972336, 18.1782055, -52.8412819, 18.2148991, -69.3122101, 69.2999878
13: -35.7285233, 38.6616020, -35.7821884, 38.6921616, -74.4206848, 74.4437866
14: -78.3675842, 11.0254440, -78.4401703, 11.0783501, -89.4459381, 89.4656143
15: -30.3212585, 30.0966797, -30.3801880, 30.1282921, -60.4495506, 60.4768677
16: -46.1951218, 30.8559685, -46.2737274, 30.9176216, -77.1000824, 77.1184082
17: -77.7688446, 14.5779533, -77.8796539, 14.7048569, -92.4737015, 92.4576111
18: -45.7925606, 21.2019806, -45.8124504, 21.2782822, -67.0708466, 67.0144348
19: -34.4013519, 10.9293413, -34.4709930, 10.9948044, -45.3961563, 45.4003334
20: -30.5384159, 14.2662916, -30.5839272, 14.3090858, -44.8475037, 44.8502197
21: -42.5536499, 14.8669319, -42.6460152, 14.9592361, -57.5128860, 57.5129471
22: -43.1313400, 17.5324516, -43.2537537, 17.6358585, -60.7671967, 60.7862053
23: -34.2711792, 15.1000576, -34.3595047, 15.1837816, -49.4549599, 49.4595642
24: -36.2930603, 14.8598251, -36.3673363, 14.9134722, -51.2065315, 51.2271614
25: -35.4157486, 17.2555008, -35.5138054, 17.3501244, -52.7658730, 52.7693062
26: -53.5136795, 20.2012787, -53.5722046, 20.2711811, -73.7848587, 73.7734833
27: -36.2062378, 18.9126644, -36.2295532, 18.9555645, -55.1618042, 55.1422195
28: -33.2059784, 18.9547539, -33.2810440, 19.0278130, -52.2337914, 52.2357979
29: -44.7293282, 16.7592945, -44.8826828, 16.8894558, -61.6187820, 61.6419754
30: -42.6847382, 19.9746418, -42.7909088, 20.0670853, -62.7518234, 62.7655487
31: -42.2232361, 15.2999964, -42.2880554, 15.3587799, -57.5820160, 57.5880508
32: -38.4778976, 23.1507893, -38.5239334, 23.1850681, -61.6629639, 61.6747208
33: -48.7968979, 35.9393921, -48.8640213, 35.9656181, -84.7625122, 84.8034134
34: -47.1310043, 21.0994854, -47.1739044, 21.1258240, -68.2184982, 68.2352982
35: -41.6537285, 26.4069748, -41.7011795, 26.4324856, -67.6192474, 67.6334763
36: -42.4311600, 26.6017456, -42.4713211, 26.6308441, -68.1909485, 68.2087326
37: -66.8107376, 22.1563873, -66.8750305, 22.2462616, -86.4938660, 86.4720306
38: -52.5088120, 31.1005630, -52.5766525, 31.1981258, -81.8597107, 81.8421631
39: -60.2186775, 35.3524628, -60.3009644, 35.4123154, -95.6309967, 95.6534271
40: -53.4357834, 28.1600418, -53.5567818, 28.2999992, -81.7357788, 81.7168274
41: -39.0460205, 27.0759315, -39.1037712, 27.1387291, -66.1847534, 66.1797028
42: -32.5021286, 21.9792099, -32.5399170, 22.0035038, -54.5056305, 54.5191269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5198652, upper bound: 37.4387754
time: 53.62 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5198652, upper bound: 37.4833486
time: 50.76 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -43.1053009, 35.1760216, -43.0998306, 35.2087822, -78.3140869, 78.2758484
1: -23.3394089, 31.9958305, -23.3457775, 32.0135612, -55.3529701, 55.3416061
2: -18.8364906, 31.9065895, -18.8373470, 31.9205437, -50.7380524, 50.7219772
3: -19.0046558, 35.1654510, -19.0148659, 35.1658630, -53.7341766, 53.7463341
4: -23.4919853, 35.9482651, -23.5024834, 35.9843788, -59.4763641, 59.4497566
5: -21.1848526, 35.4756355, -21.1916656, 35.4906387, -56.0496216, 56.0311165
6: -42.1906509, 26.0618515, -42.1848602, 26.0954819, -68.2861328, 68.2467117
7: -30.3986626, 34.2124519, -30.4109650, 34.2306976, -63.9867325, 63.9691315
8: -29.0096588, 40.0688629, -29.0173607, 40.0947723, -69.1044312, 69.0862274
9: -24.3973255, 31.6331558, -24.4069099, 31.6482582, -54.9066467, 54.8855133
10: -45.8690681, 31.3349419, -45.8765259, 31.3018112, -77.1708832, 77.2114716
11: -48.8288193, 18.2205048, -48.8734665, 18.2185287, -67.0473480, 67.0939713
12: -52.8436775, 18.2151546, -52.8504028, 18.2182732, -69.3652649, 69.3992233
13: -35.7606201, 38.6964569, -35.7800713, 38.6991920, -74.4598083, 74.4765320
14: -78.4474335, 11.1241016, -78.4690399, 11.0828934, -89.5303268, 89.5931396
15: -30.3716087, 30.1215401, -30.3926201, 30.1319752, -60.5035858, 60.5141602
16: -46.2606049, 30.9060364, -46.2826538, 30.9303513, -77.1834335, 77.1775742
17: -77.8784485, 14.7199154, -77.9173889, 14.7099628, -92.5884094, 92.6373062
18: -45.8411636, 21.2602043, -45.8227425, 21.2895927, -67.1307526, 67.0829468
19: -34.4652596, 10.9939966, -34.4923096, 10.9966183, -45.4618759, 45.4863052
20: -30.5787621, 14.3244982, -30.5981083, 14.3127327, -44.8914948, 44.9226074
21: -42.6325607, 14.9643030, -42.6747398, 14.9622307, -57.5947914, 57.6390419
22: -43.2421570, 17.6452484, -43.2970963, 17.6393318, -60.8814888, 60.9423447
23: -34.3570175, 15.1901321, -34.3933182, 15.1865263, -49.5435448, 49.5834503
24: -36.3542786, 14.9153786, -36.3887520, 14.9159184, -51.2701950, 51.3041306
25: -35.5000763, 17.3588047, -35.5499802, 17.3560066, -52.8560829, 52.9087830
26: -53.5588989, 20.2643032, -53.5846825, 20.2740841, -73.8329849, 73.8489838
27: -36.2347336, 18.9402618, -36.2361794, 18.9548988, -55.1896324, 55.1764412
28: -33.2798691, 19.0295029, -33.3070068, 19.0311565, -52.3110275, 52.3365097
29: -44.8713684, 16.8957996, -44.9395714, 16.8917160, -61.7630844, 61.8353729
30: -42.7785034, 20.0779037, -42.8283081, 20.0723286, -62.8508301, 62.9062119
31: -42.2800064, 15.3580866, -42.3070488, 15.3613329, -57.6413383, 57.6651344
32: -38.5216675, 23.1642876, -38.5262375, 23.1867867, -61.7084541, 61.6905251
33: -48.8766556, 35.9783401, -48.8728104, 35.9782639, -84.8549194, 84.8511505
34: -47.1891632, 21.1350250, -47.1806335, 21.1301098, -68.2815628, 68.2799988
35: -41.6984901, 26.4166718, -41.7051620, 26.4353027, -67.6765747, 67.6577530
36: -42.5031815, 26.6317444, -42.4758759, 26.6391563, -68.2895203, 68.2437439
37: -66.9309998, 22.2621765, -66.8860397, 22.2926064, -86.7376862, 86.6054993
38: -52.6027756, 31.1917896, -52.5828171, 31.2340450, -82.0473175, 81.9490814
39: -60.3309860, 35.4252739, -60.3084869, 35.4431000, -95.7740860, 95.7337646
40: -53.5976944, 28.3078289, -53.5648575, 28.3653965, -81.9630890, 81.8726883
41: -39.1294022, 27.1335526, -39.1109772, 27.1622658, -66.2916718, 66.2445297
42: -32.5388489, 22.0079861, -32.5465469, 22.0108299, -54.5496788, 54.5545349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 613

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4838811, upper bound: 37.5039555
time: 45.90 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4838811, upper bound: 37.5039555
time: 59.78 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -43.1005402, 35.2110748, -43.0893173, 35.1760788, -78.2766190, 78.3003922
1: -23.3461075, 32.0186958, -23.3410435, 32.0041199, -55.3502274, 55.3597412
2: -18.8369160, 31.9234390, -18.8313751, 31.9093895, -50.7259445, 50.7324371
3: -19.0110378, 35.1660843, -19.0017624, 35.1606140, -53.7494507, 53.7320175
4: -23.5004120, 35.9914970, -23.4930630, 35.9631119, -59.4619522, 59.4752121
5: -21.1914406, 35.4950333, -21.1862183, 35.4783669, -56.0392075, 56.0452538
6: -42.1851692, 26.1014824, -42.1793213, 26.0784950, -68.2636642, 68.2808075
7: -30.4119682, 34.2427177, -30.4063187, 34.2300339, -63.9938126, 63.9934845
8: -29.0181942, 40.0991592, -29.0152264, 40.0734863, -69.0916824, 69.1143875
9: -24.4089451, 31.6572037, -24.4004536, 31.6486244, -54.9357986, 54.9224625
10: -45.8569145, 31.3033028, -45.8358078, 31.2960854, -77.1529999, 77.1391144
11: -48.8733673, 18.2186451, -48.8406601, 18.2154007, -67.0887680, 67.0593033
12: -52.8535042, 18.2180805, -52.8447418, 18.2128887, -69.3696136, 69.3819427
13: -35.7782135, 38.6986465, -35.7817993, 38.6909103, -74.4691238, 74.4804459
14: -78.4586258, 11.0816956, -78.4286118, 11.0766869, -89.5353088, 89.5103073
15: -30.3969383, 30.1345291, -30.3832932, 30.1291637, -60.5261002, 60.5178223
16: -46.2826347, 30.9468269, -46.2729607, 30.9350319, -77.2090454, 77.2094269
17: -77.9183884, 14.7112503, -77.8802185, 14.7056923, -92.6240845, 92.5914688
18: -45.8177414, 21.2810669, -45.8067436, 21.2624817, -67.0802231, 67.0878143
19: -34.4974365, 10.9976168, -34.4757614, 10.9957743, -45.4932098, 45.4733772
20: -30.5990467, 14.3117390, -30.5850430, 14.3078756, -44.9069214, 44.8967819
21: -42.6814537, 14.9617929, -42.6524925, 14.9585600, -57.6400146, 57.6142845
22: -43.3017426, 17.6393356, -43.2577972, 17.6356316, -60.9373741, 60.8971329
23: -34.3935890, 15.1862974, -34.3581810, 15.1832447, -49.5768356, 49.5444794
24: -36.4001465, 14.9156084, -36.3786736, 14.9129066, -51.3130531, 51.2942810
25: -35.5560341, 17.3546257, -35.5196533, 17.3481522, -52.9041862, 52.8742790
26: -53.5955429, 20.2744637, -53.5842628, 20.2710896, -73.8666306, 73.8587265
27: -36.2372360, 18.9518566, -36.2300262, 18.9539490, -55.1911850, 55.1818848
28: -33.3079605, 19.0304661, -33.2768784, 19.0269012, -52.3348618, 52.3073425
29: -44.9456558, 16.8908234, -44.8879204, 16.8891106, -61.8347664, 61.7787437
30: -42.8305664, 20.0702934, -42.7894325, 20.0644302, -62.8949966, 62.8597260
31: -42.3122101, 15.3623457, -42.2927780, 15.3596449, -57.6718559, 57.6551247
32: -38.5308762, 23.2036400, -38.5242310, 23.2022781, -61.7331543, 61.7278709
33: -48.8640366, 35.9706535, -48.8627625, 35.9568939, -84.8209305, 84.8334198
34: -47.1773453, 21.1304035, -47.1684036, 21.1266766, -68.2661285, 68.2601013
35: -41.7111092, 26.4367371, -41.7074089, 26.4342670, -67.6728973, 67.6813354
36: -42.4757195, 26.6321983, -42.4714165, 26.6239357, -68.2375946, 68.2322998
37: -66.8859863, 22.2576332, -66.8747253, 22.2075748, -86.5536346, 86.5567780
38: -52.5813065, 31.2141590, -52.5760422, 31.1782913, -81.9419403, 81.9515533
39: -60.3079529, 35.4285889, -60.3010445, 35.3948212, -95.7027740, 95.7296295
40: -53.5655365, 28.3525734, -53.5569572, 28.2868633, -81.8524017, 81.9095306
41: -39.1093674, 27.1609650, -39.1015396, 27.1371994, -66.2465668, 66.2625046
42: -32.5419312, 22.0088539, -32.5342026, 22.0006599, -54.5425911, 54.5430565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4578614, upper bound: 37.5198652
time: 54.50 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4578614, upper bound: 37.5198652
time: 52.22 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -43.1103325, 35.2449112, -43.1947594, 35.2621994, -78.3725281, 78.4396667
1: -23.3509140, 32.0343018, -23.3931007, 32.0476685, -55.3985825, 55.4274025
2: -18.8418522, 31.9363174, -18.8821640, 31.9480362, -50.7693176, 50.8007851
3: -19.0141525, 35.1718636, -19.0315323, 35.1883316, -53.7802963, 53.7705307
4: -23.5072441, 36.0122185, -23.5612240, 36.0243073, -59.5295105, 59.5734406
5: -21.1953545, 35.5117416, -21.2196426, 35.5333099, -56.0982437, 56.1106987
6: -42.1907196, 26.1250706, -42.2426682, 26.1422501, -68.3329697, 68.3677368
7: -30.4166679, 34.2543297, -30.4541492, 34.2736206, -64.0438995, 64.0750580
8: -29.0209694, 40.1244049, -29.0735435, 40.1492958, -69.1702652, 69.1979523
9: -24.4109917, 31.6665421, -24.4365444, 31.6815891, -54.9559097, 54.9798698
10: -45.8788910, 31.3101997, -45.9114761, 31.3765469, -77.2554398, 77.2216797
11: -48.9055939, 18.2219105, -48.9252281, 18.3135796, -67.2191772, 67.1471405
12: -52.8627472, 18.2215538, -52.8913193, 18.2514267, -69.4704285, 69.4357834
13: -35.7760925, 38.7058334, -35.8140411, 38.7266693, -74.5027618, 74.5198746
14: -78.4876862, 11.0863266, -78.5086746, 11.1762466, -89.6639328, 89.5950012
15: -30.4094315, 30.1383476, -30.4341316, 30.1546288, -60.5640602, 60.5724792
16: -46.2915955, 30.9596119, -46.3386765, 30.9851265, -77.2681808, 77.2930603
17: -77.9562378, 14.7164135, -77.9902267, 14.8483753, -92.8046112, 92.7066422
18: -45.8283768, 21.2944183, -45.8613739, 21.3240013, -67.1523743, 67.1557922
19: -34.5188637, 10.9994354, -34.5400543, 11.0606022, -45.5794678, 45.5394897
20: -30.6133099, 14.3153982, -30.6255054, 14.3667870, -44.9800949, 44.9409027
21: -42.7103043, 14.9647989, -42.7332573, 15.0562391, -57.7665443, 57.6980553
22: -43.3451271, 17.6428833, -43.3688316, 17.7487717, -61.0938988, 61.0117149
23: -34.4274445, 15.1891127, -34.4454193, 15.2740774, -49.7015228, 49.6345329
24: -36.4216385, 14.9180832, -36.4400673, 14.9686050, -51.3902435, 51.3581505
25: -35.5922394, 17.3607635, -35.6040878, 17.4524422, -53.0446815, 52.9648514
26: -53.6080933, 20.2773285, -53.6297226, 20.3342075, -73.9422989, 73.9070511
27: -36.2440033, 18.9512024, -36.2590790, 18.9818707, -55.2258759, 55.2102814
28: -33.3339844, 19.0338764, -33.3515549, 19.1022110, -52.4361954, 52.3854294
29: -45.0025330, 16.8933067, -45.0300331, 17.0261345, -62.0286674, 61.9233398
30: -42.8684311, 20.0756264, -42.8855553, 20.1685715, -63.0370026, 62.9611816
31: -42.3313675, 15.3648996, -42.3498192, 15.4180574, -57.7494240, 57.7147179
32: -38.5333252, 23.2053795, -38.5685463, 23.2160568, -61.7493820, 61.7739258
33: -48.8727951, 35.9834137, -48.9430084, 35.9961014, -84.8688965, 84.9264221
34: -47.1841736, 21.1346836, -47.2268066, 21.1619339, -68.3108978, 68.3236313
35: -41.7150955, 26.4396229, -41.7525253, 26.4466171, -67.6990509, 67.7394104
36: -42.4803314, 26.6411457, -42.5460281, 26.6546459, -68.2730408, 68.3318863
37: -66.8970642, 22.3076878, -66.9995651, 22.3184910, -86.6873703, 86.8017654
38: -52.5877533, 31.2547550, -52.6757278, 31.2766075, -82.0563965, 82.1470490
39: -60.3156700, 35.4593506, -60.4140778, 35.4676666, -95.7833405, 95.8734283
40: -53.5737305, 28.4179802, -53.7205429, 28.4367390, -82.0104675, 82.1385193
41: -39.1166687, 27.1846027, -39.1859512, 27.1950111, -66.3116760, 66.3705521
42: -32.5494385, 22.0163670, -32.5719223, 22.0308342, -54.5802727, 54.5882874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 613

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5073247, upper bound: 37.4838810
time: 54.92 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5073247, upper bound: 37.5260364
time: 50.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 107.34 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 107.34
Output dim: 8, lower bound: -37.5198652, upper bound: 37.4387754
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 107.34
Output dim: 8, lower bound: -37.5198652, upper bound: 37.4833486
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 107.34
Output dim: 8, lower bound: -37.4838811, upper bound: 37.5039555
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 107.34
Output dim: 8, lower bound: -37.4838811, upper bound: 37.5039555
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 107.34
Output dim: 8, lower bound: -37.4578614, upper bound: 37.5198652
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 107.34
Output dim: 8, lower bound: -37.4578614, upper bound: 37.5198652
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 107.34
Output dim: 8, lower bound: -37.5073247, upper bound: 37.4838810
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 107.34
Output dim: 8, lower bound: -37.5073247, upper bound: 37.5260364

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -42.8734474, 34.9982758, -43.0625305, 35.1472168, -78.0206604, 78.0608063
1: -23.2165394, 31.8766441, -23.3277321, 31.9789143, -55.1954536, 55.2043762
2: -18.6739006, 31.7533684, -18.8168259, 31.8603115, -50.5094337, 50.5476494
3: -18.8270607, 34.9610367, -18.9968681, 35.0875549, -53.4639511, 53.5186462
4: -23.3340244, 35.8260765, -23.4680119, 35.9500389, -59.2752533, 59.2875824
5: -20.9868889, 35.2306938, -21.1716003, 35.3983612, -55.7375641, 55.7660980
6: -42.0556946, 25.8361206, -42.1669655, 26.0100613, -68.0657578, 68.0030823
7: -30.1773777, 33.9811859, -30.3850918, 34.1480103, -63.6524658, 63.7075119
8: -28.8465271, 39.8827667, -29.0020256, 40.0329323, -68.8794556, 68.8847961
9: -24.1974449, 31.4642105, -24.3404579, 31.6283550, -54.6721764, 54.6403008
10: -45.5185852, 31.0132484, -45.7369499, 31.2746220, -76.7932053, 76.7501984
11: -48.6703796, 17.9956245, -48.8152466, 18.1932411, -66.8636169, 66.8108673
12: -52.4082108, 17.8532887, -52.6710091, 18.1926041, -68.8919373, 68.7928772
13: -35.5613327, 38.5318985, -35.7284546, 38.6747017, -74.2360382, 74.2603531
14: -78.0153885, 10.8123226, -78.2979736, 11.0703926, -89.0857849, 89.1102982
15: -30.0819397, 29.9300041, -30.2887497, 30.1141663, -60.1961060, 60.2187538
16: -46.1324997, 30.7257595, -46.2400970, 30.8861332, -77.0058136, 76.9539795
17: -77.5095215, 14.3391762, -77.7656631, 14.6896992, -92.1992188, 92.1048431
18: -45.6604233, 21.0961094, -45.7664948, 21.2620697, -66.9224930, 66.8626022
19: -34.3308067, 10.8903866, -34.4488144, 10.9881868, -45.3189926, 45.3392029
20: -30.4414234, 14.2146139, -30.5566978, 14.2997265, -44.7411499, 44.7713127
21: -42.4782600, 14.8003111, -42.6216087, 14.9487505, -57.4270096, 57.4219208
22: -42.8830490, 17.3536339, -43.1560249, 17.6232624, -60.5063095, 60.5096588
23: -34.1864395, 15.0227394, -34.3383789, 15.1575480, -49.3439865, 49.3611183
24: -36.2038803, 14.7925510, -36.3500824, 14.8908052, -51.0946846, 51.1426315
25: -35.3252716, 17.1791000, -35.4857368, 17.3358002, -52.6610718, 52.6648369
26: -53.1519051, 19.9304504, -53.4178886, 20.2502537, -73.4021606, 73.3483429
27: -36.0887489, 18.8291340, -36.2133026, 18.9258728, -55.0146217, 55.0424347
28: -33.1097794, 18.8617287, -33.2646561, 18.9940491, -52.1038284, 52.1263847
29: -44.5516586, 16.5891457, -44.8181267, 16.8788605, -61.4305191, 61.4072723
30: -42.5228195, 19.7845078, -42.7707329, 19.9958057, -62.5186234, 62.5552406
31: -42.1761360, 15.2353821, -42.2668304, 15.3457384, -57.5218735, 57.5022125
32: -38.3564949, 23.0621452, -38.4862442, 23.1670132, -61.5235062, 61.5483894
33: -48.6458130, 35.8454132, -48.8388176, 35.9384689, -84.5842819, 84.6842346
34: -47.0075073, 20.9549274, -47.1581421, 21.0706596, -68.0378723, 68.0747910
35: -41.5157280, 26.2762737, -41.6825256, 26.3789330, -67.4170151, 67.4839706
36: -42.3410873, 26.5596695, -42.4534035, 26.6147270, -68.0819092, 68.1511993
37: -66.6725464, 22.0835724, -66.8383026, 22.2296028, -86.3252716, 86.3547363
38: -52.4089737, 31.0071392, -52.5623131, 31.1666775, -81.7123108, 81.7331238
39: -60.0613022, 35.2582779, -60.2557602, 35.3907433, -95.4520416, 95.5140381
40: -53.2874451, 28.0541840, -53.5346603, 28.2559185, -81.5433655, 81.5888443
41: -38.9784317, 26.9664459, -39.0904083, 27.1001625, -66.0785980, 66.0568542
42: -32.4486465, 21.8673248, -32.5272408, 21.9705009, -54.4191475, 54.3945656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 984

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5161551, upper bound: 37.3586012
time: 52.45 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5161551, upper bound: 37.3586012
time: 58.22 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -42.9939423, 35.0684433, -43.0859680, 35.1637917, -78.1577301, 78.1544113
1: -23.2832718, 31.9388752, -23.3386612, 31.9909992, -55.2742691, 55.2775345
2: -18.7817497, 31.8624878, -18.8302402, 31.9050140, -50.6631470, 50.6688080
3: -18.9704361, 35.1173935, -19.0093117, 35.1500320, -53.6802597, 53.6764793
4: -23.4115524, 35.8841171, -23.4893131, 35.9619484, -59.3662186, 59.3726654
5: -21.1471500, 35.4019928, -21.1853752, 35.4629707, -55.9697266, 55.9449654
6: -42.1231461, 25.9627094, -42.1768570, 26.0547543, -68.1779022, 68.1395645
7: -30.3451900, 34.1409683, -30.4031525, 34.2041550, -63.8841553, 63.8790741
8: -28.9456882, 39.9896393, -29.0114098, 40.0675278, -69.0132141, 69.0010529
9: -24.3534451, 31.5961742, -24.4008675, 31.6367798, -54.8328667, 54.8741760
10: -45.7833481, 31.2502327, -45.8495560, 31.2922688, -77.0756149, 77.0997925
11: -48.7379150, 18.0720730, -48.8387794, 18.1889381, -66.9268494, 66.9108505
12: -52.7834282, 18.1727962, -52.8345718, 18.2121029, -69.2683868, 69.2871323
13: -35.7124138, 38.6550102, -35.7729492, 38.6888580, -74.4012756, 74.4279633
14: -78.3524017, 11.0223007, -78.4324341, 11.0768604, -89.4292603, 89.4547348
15: -30.2735386, 30.0916824, -30.3539600, 30.1257401, -60.3992767, 60.4456406
16: -46.1878586, 30.7990265, -46.2699509, 30.8884850, -77.0624008, 77.0572662
17: -77.7484589, 14.5718765, -77.8696747, 14.7017975, -92.4502563, 92.4415512
18: -45.7735023, 21.1970119, -45.8028336, 21.2757301, -67.0492325, 66.9998474
19: -34.3962784, 10.9246407, -34.4684067, 10.9925022, -45.3887787, 45.3930473
20: -30.5331154, 14.2625446, -30.5811787, 14.3072214, -44.8403358, 44.8437233
21: -42.5478287, 14.8552713, -42.6430511, 14.9535789, -57.5014076, 57.4983215
22: -43.0967522, 17.5274620, -43.2315750, 17.6332874, -60.7300415, 60.7590370
23: -34.2671585, 15.0847969, -34.3574600, 15.1763258, -49.4434853, 49.4422569
24: -36.2870483, 14.8489761, -36.3643646, 14.9074841, -51.1945343, 51.2133408
25: -35.3983765, 17.2506561, -35.5053711, 17.3476295, -52.7460060, 52.7560272
26: -53.4878082, 20.1961403, -53.5595779, 20.2685013, -73.7563095, 73.7557220
27: -36.2010536, 18.8936977, -36.2269211, 18.9448318, -55.1458855, 55.1206207
28: -33.2022171, 18.9465961, -33.2790718, 19.0236740, -52.2258911, 52.2256699
29: -44.7204437, 16.7551727, -44.8782616, 16.8873329, -61.6077766, 61.6334343
30: -42.6771126, 19.9646492, -42.7870255, 20.0620804, -62.7391930, 62.7516747
31: -42.2177582, 15.2793312, -42.2852173, 15.3488007, -57.5665588, 57.5645485
32: -38.4716034, 23.1447411, -38.5207291, 23.1820831, -61.6536865, 61.6654701
33: -48.7889099, 35.9328995, -48.8599739, 35.9622307, -84.7511444, 84.7928772
34: -47.1264610, 21.0926628, -47.1716461, 21.1223373, -68.2115631, 68.2257233
35: -41.6476517, 26.4001541, -41.6981544, 26.4290886, -67.6170044, 67.6205826
36: -42.4218597, 26.5982590, -42.4667969, 26.6291046, -68.1890564, 68.2000732
37: -66.7910461, 22.1517849, -66.8653641, 22.2438049, -86.4685059, 86.4557571
38: -52.5022316, 31.0936584, -52.5734024, 31.1946507, -81.8500977, 81.8273163
39: -60.1969948, 35.3474846, -60.2902832, 35.4096870, -95.6066818, 95.6377716
40: -53.4293976, 28.1494007, -53.5534515, 28.2946606, -81.7240601, 81.7028503
41: -39.0417023, 27.0564556, -39.1015739, 27.1292686, -66.1709747, 66.1580276
42: -32.4976349, 21.9652023, -32.5376358, 21.9966183, -54.4942551, 54.5028381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 984

## Relational analysis of IS_A1_A1_A2_A1

### Relational analysis result of IS_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4398975, upper bound: 37.4796551
time: 55.29 seconds

## Relational analysis of IS_A1_A1_A2_A2

### Relational analysis result of IS_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5193923, upper bound: 37.4828745
time: 61.02 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -43.0783272, 35.0899849, -43.0009346, 35.0126228, -78.0909500, 78.0909195
1: -23.3236313, 31.9458771, -23.2854118, 31.9009037, -55.2245331, 55.2312889
2: -18.8166676, 31.8478241, -18.7623844, 31.7864647, -50.5831299, 50.5854568
3: -18.9899178, 35.1370316, -18.9740562, 35.0969391, -53.6441803, 53.6631966
4: -23.4659729, 35.8686714, -23.4230709, 35.8053169, -59.2712898, 59.2796974
5: -21.1674843, 35.3977280, -21.1233063, 35.3088646, -55.8475189, 55.8754234
6: -42.1752930, 26.0061779, -42.1346054, 25.9670353, -68.1423264, 68.1407852
7: -30.3775845, 34.1356888, -30.3225040, 34.0518265, -63.7823639, 63.7901688
8: -28.9948921, 40.0021133, -28.9524956, 39.9387970, -68.9336853, 68.9546051
9: -24.3744659, 31.6067848, -24.3475666, 31.5872231, -54.8341179, 54.8045311
10: -45.7731781, 31.3073997, -45.6517029, 31.1857643, -76.9589386, 76.9591064
11: -48.7587128, 18.2037086, -48.7153854, 18.1474152, -66.9061279, 66.9190979
12: -52.7345276, 18.1943703, -52.5949478, 18.1142197, -69.1160736, 69.1084518
13: -35.7369690, 38.6671600, -35.7140350, 38.6302185, -74.3671875, 74.3811951
14: -78.3107071, 11.1097603, -78.1510086, 10.9791279, -89.2898331, 89.2607727
15: -30.2970943, 30.1059856, -30.2167053, 30.0757217, -60.3728180, 60.3226929
16: -46.2363663, 30.8304672, -46.2012024, 30.7512169, -76.9794617, 77.0180359
17: -77.7377625, 14.7026939, -77.5904694, 14.5866489, -92.3244095, 92.2931671
18: -45.8108711, 21.2320671, -45.7955208, 21.2255707, -67.0364380, 67.0275879
19: -34.4258499, 10.9865952, -34.3958321, 10.9624815, -45.3883324, 45.3824272
20: -30.5386047, 14.3128557, -30.5013313, 14.2658701, -44.8044739, 44.8141861
21: -42.5746727, 14.9521475, -42.5350418, 14.9025183, -57.4771919, 57.4871902
22: -43.1206436, 17.6315517, -43.0159073, 17.5464745, -60.6671181, 60.6474609
23: -34.2904091, 15.1785355, -34.2389450, 15.1316137, -49.4220238, 49.4174805
24: -36.3128586, 14.9064407, -36.2932968, 14.8781204, -51.1909790, 51.1997375
25: -35.4229050, 17.3386002, -35.3729630, 17.2722969, -52.6952019, 52.7115631
26: -53.4537277, 20.2506962, -53.3389740, 20.1853676, -73.6390991, 73.5896683
27: -36.2135963, 18.9219532, -36.1943970, 18.9105129, -55.1241074, 55.1163483
28: -33.2203102, 19.0154610, -33.1700974, 18.9965668, -52.2168770, 52.1855583
29: -44.7346344, 16.8838615, -44.6230240, 16.7947559, -61.5293884, 61.5068855
30: -42.7084732, 20.0574875, -42.6896210, 20.0078964, -62.7163696, 62.7471085
31: -42.2423019, 15.3474751, -42.2161331, 15.3204126, -57.5627136, 57.5636063
32: -38.4942627, 23.1512299, -38.4586449, 23.1665840, -61.6608467, 61.6098747
33: -48.8526917, 35.9341125, -48.7905083, 35.8735771, -84.7262726, 84.7246246
34: -47.1638107, 21.1158752, -47.1114922, 21.1011162, -68.2275162, 68.1915436
35: -41.6808014, 26.4011288, -41.6533585, 26.3973160, -67.6151886, 67.5834427
36: -42.4846420, 26.6139107, -42.4115067, 26.5954227, -68.2153320, 68.1447296
37: -66.9005127, 22.1475029, -66.8194733, 22.0489407, -86.4206390, 86.3674088
38: -52.5792389, 31.1179619, -52.5234985, 31.0725574, -81.8383331, 81.7809372
39: -60.3021469, 35.3412094, -60.2234001, 35.2567444, -95.5588913, 95.5646057
40: -53.5775604, 28.1819286, -53.4714813, 28.0814953, -81.6590576, 81.6534119
41: -39.1097717, 27.0684643, -39.0412521, 27.0132561, -66.1230316, 66.1097183
42: -32.5034714, 21.9772129, -32.4675446, 21.9729404, -54.4764099, 54.4447556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4762770, upper bound: 37.4512250
time: 52.91 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4762770, upper bound: 37.4951790
time: 56.67 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -43.1044159, 35.1752472, -43.0954781, 35.2051468, -78.3095627, 78.2707214
1: -23.3388157, 31.9952412, -23.3429298, 32.0108261, -55.3496399, 55.3381729
2: -18.8358498, 31.9059639, -18.8341770, 31.9175835, -50.7343445, 50.7192345
3: -19.0040264, 35.1649323, -19.0117645, 35.1633873, -53.7299652, 53.7502480
4: -23.4913597, 35.9472198, -23.4994011, 35.9782753, -59.4696350, 59.4466209
5: -21.1841640, 35.4749374, -21.1883717, 35.4872360, -56.0450058, 56.0313492
6: -42.1901398, 26.0611820, -42.1822052, 26.0923004, -68.2824402, 68.2433853
7: -30.3980083, 34.2117538, -30.4077511, 34.2273712, -63.9821243, 63.9707642
8: -29.0089569, 40.0682259, -29.0139046, 40.0917397, -69.1006927, 69.0821304
9: -24.3961468, 31.6322060, -24.4011593, 31.6437912, -54.8985634, 54.8738251
10: -45.8681946, 31.3337822, -45.8722763, 31.2961655, -77.1643600, 77.2060547
11: -48.8270264, 18.2199097, -48.8639984, 18.2155437, -67.0425720, 67.0839081
12: -52.8427620, 18.2142124, -52.8457985, 18.2136745, -69.3730164, 69.3919983
13: -35.7600136, 38.6953392, -35.7770844, 38.6937866, -74.4537964, 74.4724274
14: -78.4461517, 11.1235771, -78.4630432, 11.0805016, -89.5266571, 89.5866241
15: -30.3704605, 30.1210938, -30.3873062, 30.1297874, -60.5002480, 60.5084000
16: -46.2598991, 30.9044724, -46.2791557, 30.9228306, -77.1750641, 77.1736908
17: -77.8771973, 14.7192898, -77.9113617, 14.7069511, -92.5841522, 92.6306534
18: -45.8405418, 21.2568607, -45.8197746, 21.2730484, -67.1135864, 67.0766373
19: -34.4647293, 10.9935446, -34.4897270, 10.9943562, -45.4590836, 45.4832726
20: -30.5782528, 14.3241119, -30.5956535, 14.3109055, -44.8891602, 44.9197655
21: -42.6318436, 14.9637985, -42.6712494, 14.9597788, -57.5916214, 57.6350479
22: -43.2409859, 17.6446133, -43.2913399, 17.6362305, -60.8772163, 60.9359512
23: -34.3563042, 15.1896887, -34.3899994, 15.1843615, -49.5406647, 49.5796890
24: -36.3534927, 14.9151478, -36.3850403, 14.9148331, -51.2683258, 51.3001862
25: -35.4991989, 17.3583546, -35.5458450, 17.3538570, -52.8530579, 52.9041977
26: -53.5577736, 20.2635212, -53.5792122, 20.2703056, -73.8280792, 73.8427353
27: -36.2341232, 18.9392681, -36.2331963, 18.9496307, -55.1837540, 55.1724625
28: -33.2787895, 19.0290489, -33.3017349, 19.0290623, -52.3078537, 52.3307838
29: -44.8700562, 16.8951950, -44.9332504, 16.8887291, -61.7587852, 61.8284454
30: -42.7776489, 20.0773315, -42.8246613, 20.0696297, -62.8472786, 62.9019928
31: -42.2792549, 15.3576584, -42.3034973, 15.3591890, -57.6384430, 57.6611557
32: -38.5201111, 23.1637993, -38.5194855, 23.1844368, -61.7045479, 61.6832848
33: -48.8760071, 35.9777870, -48.8697281, 35.9757233, -84.8517303, 84.8475189
34: -47.1875381, 21.1345310, -47.1725998, 21.1278191, -68.2773209, 68.2712708
35: -41.6979713, 26.4161892, -41.7027016, 26.4332981, -67.6658936, 67.6547546
36: -42.5026627, 26.6310120, -42.4733429, 26.6356335, -68.2753983, 68.2405014
37: -66.9302826, 22.2608681, -66.8826599, 22.2864094, -86.6989288, 86.6006241
38: -52.6020470, 31.1897507, -52.5790787, 31.2237549, -82.0164566, 81.9433289
39: -60.3302002, 35.4241943, -60.3048668, 35.4378471, -95.7680511, 95.7290649
40: -53.5970230, 28.3067474, -53.5616989, 28.3607845, -81.9578094, 81.8684464
41: -39.1288834, 27.1328049, -39.1084251, 27.1586685, -66.2875519, 66.2412262
42: -32.5368919, 22.0073795, -32.5378342, 22.0079231, -54.5448151, 54.5452118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 613

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5172994, upper bound: 37.4512250
time: 48.36 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5172994, upper bound: 37.4951790
time: 58.53 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -43.0729561, 35.1832275, -42.9606590, 35.0825577, -78.1555176, 78.1438904
1: -23.3328419, 31.9995918, -23.2695827, 31.9283390, -55.2611809, 55.2691727
2: -18.8212166, 31.8760662, -18.7189217, 31.7947502, -50.5947876, 50.5714188
3: -18.9961281, 35.0934258, -18.8534908, 34.9838638, -53.5538406, 53.4998055
4: -23.4724903, 35.9774017, -23.4020386, 35.9007416, -59.3657227, 59.3651161
5: -21.1751556, 35.4193573, -21.0209656, 35.2882652, -55.8329315, 55.7978897
6: -42.1727219, 26.0394878, -42.1068954, 25.9162712, -68.0889893, 68.1463852
7: -30.3906403, 34.1715813, -30.2320251, 34.0423393, -63.7822037, 63.7398148
8: -29.0055370, 40.0618668, -28.9096069, 39.9612770, -68.9668121, 68.9714737
9: -24.3444977, 31.6464844, -24.2365265, 31.5120869, -54.7100601, 54.7448158
10: -45.7389832, 31.2829456, -45.5601501, 31.0538330, -76.7928162, 76.8430939
11: -48.8454819, 18.1964760, -48.7645721, 18.0856743, -66.9311523, 66.9610443
12: -52.6832848, 18.1958237, -52.4557724, 17.8879509, -68.8625183, 68.9617386
13: -35.7244911, 38.6811752, -35.6146164, 38.5611801, -74.2856750, 74.2957916
14: -78.3163910, 11.0737591, -78.0764008, 10.8635635, -89.1799545, 89.1501617
15: -30.3054810, 30.1204147, -30.1440639, 29.9624634, -60.2679443, 60.2644806
16: -46.2490158, 30.9153824, -46.2102623, 30.8048172, -77.0446320, 77.1151505
17: -77.8043671, 14.6960888, -77.6209564, 14.4669151, -92.2712860, 92.3170471
18: -45.7717896, 21.2648544, -45.6744843, 21.1565170, -66.9283066, 66.9393387
19: -34.4752274, 10.9910011, -34.4052429, 10.9568491, -45.4320755, 45.3962440
20: -30.5717888, 14.3023872, -30.4880352, 14.2561646, -44.8279533, 44.7904205
21: -42.6570511, 14.9512653, -42.5771255, 14.8919678, -57.5490189, 57.5283890
22: -43.2040520, 17.6267433, -43.0096054, 17.4568081, -60.6608582, 60.6363487
23: -34.3724785, 15.1600752, -34.2734070, 15.1059370, -49.4784164, 49.4334831
24: -36.3829002, 14.8929558, -36.2894363, 14.8456297, -51.2285309, 51.1823921
25: -35.5279579, 17.3402863, -35.4291725, 17.2717476, -52.7997055, 52.7694588
26: -53.4412155, 20.2535038, -53.2225342, 20.0002022, -73.4414215, 73.4760361
27: -36.2210083, 18.9221592, -36.1125259, 18.8704262, -55.0914345, 55.0346832
28: -33.2915802, 18.9966793, -33.1806870, 18.9338989, -52.2254791, 52.1773682
29: -44.8810844, 16.8802738, -44.7103958, 16.7189941, -61.6000786, 61.5906677
30: -42.8103485, 19.9990082, -42.6274643, 19.8741570, -62.6845055, 62.6264725
31: -42.2910233, 15.3492880, -42.2456055, 15.2950335, -57.5860558, 57.5948944
32: -38.4931984, 23.1855583, -38.4028931, 23.1136379, -61.6068344, 61.5884514
33: -48.8388100, 35.9435501, -48.7116318, 35.8629990, -84.7018127, 84.6551819
34: -47.1616020, 21.0752068, -47.0448761, 20.9822044, -68.1056824, 68.0794601
35: -41.6924210, 26.3831844, -41.5693817, 26.3035469, -67.5234222, 67.4791336
36: -42.4578171, 26.6160984, -42.3810272, 26.5818119, -68.1800079, 68.1229553
37: -66.8492432, 22.2409420, -66.7365112, 22.1347046, -86.4363098, 86.3881683
38: -52.5669861, 31.1827087, -52.4762268, 31.0849323, -81.8330078, 81.8038635
39: -60.2627563, 35.4070587, -60.1437531, 35.3005943, -95.5633545, 95.5508118
40: -53.5434189, 28.3084850, -53.4086304, 28.1812077, -81.7246246, 81.7171173
41: -39.0960121, 27.1223850, -39.0339661, 27.0277233, -66.1237335, 66.1563492
42: -32.5292397, 21.9758224, -32.4806938, 21.8887444, -54.4179840, 54.4565163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 984

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3777108, upper bound: 37.5161551
time: 52.32 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4573876, upper bound: 37.5193925
time: 54.41 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -43.0963707, 35.1997948, -43.0811920, 35.1527328, -78.2490997, 78.2809906
1: -23.3437805, 32.0116959, -23.3363514, 31.9905853, -55.3343658, 55.3480453
2: -18.8346214, 31.9207573, -18.8267765, 31.9038639, -50.7159882, 50.7251587
3: -19.0085583, 35.1559181, -18.9969101, 35.1402473, -53.7116623, 53.7161102
4: -23.4938297, 35.9893303, -23.4795990, 35.9587250, -59.4507675, 59.4561806
5: -21.1889515, 35.4840088, -21.1812916, 35.4595490, -56.0117569, 56.0300674
6: -42.1826057, 26.0842190, -42.1743202, 26.0427799, -68.2253876, 68.2585373
7: -30.4087143, 34.2277374, -30.3998699, 34.2021217, -63.9536362, 63.9716492
8: -29.0149498, 40.0964699, -29.0087891, 40.0681190, -69.0830688, 69.1052551
9: -24.4048939, 31.6549377, -24.3925056, 31.6440487, -54.9439621, 54.9055290
10: -45.8515930, 31.3006077, -45.8248825, 31.2907677, -77.1423645, 77.1254883
11: -48.8690453, 18.1921692, -48.8321533, 18.1623039, -67.0313492, 67.0243225
12: -52.8468552, 18.2152882, -52.8309822, 18.2074852, -69.3567505, 69.3381348
13: -35.7689781, 38.6953278, -35.7656784, 38.6843033, -74.4532776, 74.4610062
14: -78.4509048, 11.0802040, -78.4134293, 11.0735550, -89.5244598, 89.4936371
15: -30.3706913, 30.1319675, -30.3355560, 30.1241570, -60.4948502, 60.4675217
16: -46.2788620, 30.9177322, -46.2656937, 30.8780842, -77.1479034, 77.1717987
17: -77.9084015, 14.7082157, -77.8597794, 14.6996078, -92.6080093, 92.5679932
18: -45.8081360, 21.2784920, -45.7876587, 21.2574844, -67.0656204, 67.0661469
19: -34.4948273, 10.9952927, -34.4706841, 10.9911041, -45.4859314, 45.4659767
20: -30.5962811, 14.3098679, -30.5797310, 14.3041191, -44.9003983, 44.8895988
21: -42.6784821, 14.9561577, -42.6466331, 14.9469481, -57.6254311, 57.6027908
22: -43.2795792, 17.6367722, -43.2231789, 17.6306267, -60.9102058, 60.8599510
23: -34.3915482, 15.1788664, -34.3541718, 15.1680002, -49.5595474, 49.5330391
24: -36.3971672, 14.9096222, -36.3726387, 14.9020519, -51.2992172, 51.2822609
25: -35.5476074, 17.3521233, -35.5023003, 17.3433304, -52.8909378, 52.8544235
26: -53.5828972, 20.2717705, -53.5583954, 20.2659397, -73.8488388, 73.8301697
27: -36.2346230, 18.9411392, -36.2248230, 18.9349899, -55.1696129, 55.1659622
28: -33.3060226, 19.0262947, -33.2731171, 19.0187607, -52.3247833, 52.2994118
29: -44.9412422, 16.8887215, -44.8790665, 16.8849640, -61.8262062, 61.7677879
30: -42.8266754, 20.0652657, -42.7817764, 20.0544395, -62.8811150, 62.8470421
31: -42.3093834, 15.3523455, -42.2873001, 15.3390398, -57.6484222, 57.6396446
32: -38.5276794, 23.2006454, -38.5179138, 23.1962128, -61.7238922, 61.7185593
33: -48.8599815, 35.9673500, -48.8547668, 35.9504852, -84.8104706, 84.8221130
34: -47.1750870, 21.1269302, -47.1638412, 21.1198654, -68.2565689, 68.2531357
35: -41.7080345, 26.4333153, -41.7013092, 26.4274292, -67.6599731, 67.6790924
36: -42.4712219, 26.6304665, -42.4621391, 26.6204643, -68.2289429, 68.2304230
37: -66.8763123, 22.2552414, -66.8550186, 22.2029419, -86.5373001, 86.5314407
38: -52.5780640, 31.2106857, -52.5694885, 31.1714077, -81.9271698, 81.9419937
39: -60.2973022, 35.4259987, -60.2793808, 35.3898201, -95.6871185, 95.7053833
40: -53.5621796, 28.3472004, -53.5505867, 28.2762108, -81.8383942, 81.8977890
41: -39.1071663, 27.1514816, -39.0972214, 27.1177673, -66.2249298, 66.2487030
42: -32.5396576, 22.0019569, -32.5296936, 21.9866543, -54.5263138, 54.5316505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 984

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4540900, upper bound: 37.4398974
time: 57.81 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5021107, upper bound: 37.5193922
time: 46.63 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -43.0114784, 35.0486526, -43.1678314, 35.1761742, -78.1876526, 78.2164841
1: -23.2906189, 31.9216232, -23.3773689, 31.9976768, -55.2882957, 55.2989922
2: -18.7669125, 31.8022575, -18.8623943, 31.8892708, -50.6328506, 50.6458931
3: -18.9733658, 35.1028900, -19.0166969, 35.1598740, -53.6972046, 53.6803970
4: -23.4279137, 35.8331299, -23.5351849, 35.9446907, -59.3595581, 59.3655815
5: -21.1270199, 35.3299332, -21.2022800, 35.4554405, -55.9425507, 55.9085846
6: -42.1404572, 25.9964657, -42.2272987, 26.0865097, -68.2269669, 68.2237625
7: -30.3282356, 34.0754585, -30.4331131, 34.1969261, -63.8650131, 63.8706360
8: -28.9560909, 39.9683571, -29.0587349, 40.0825272, -69.0386200, 69.0270920
9: -24.3515644, 31.6054535, -24.4138107, 31.6551285, -54.8747559, 54.9075394
10: -45.6540222, 31.1942616, -45.8156395, 31.3491096, -77.0031281, 77.0099030
11: -48.7474670, 18.1507950, -48.8551064, 18.2967758, -67.0442429, 67.0059052
12: -52.6072998, 18.1174774, -52.7821198, 18.2306595, -69.1796646, 69.1866455
13: -35.7101059, 38.6368408, -35.7902603, 38.6974716, -74.4075775, 74.4271011
14: -78.1696091, 10.9826050, -78.3719406, 11.1619892, -89.3315964, 89.3545456
15: -30.2334614, 30.0821190, -30.3595638, 30.1390991, -60.3725586, 60.4416809
16: -46.2101898, 30.7804642, -46.3144684, 30.9095116, -77.1086960, 77.0890961
17: -77.6292725, 14.5932121, -77.8495483, 14.8312168, -92.4604874, 92.4427643
18: -45.8011513, 21.2283039, -45.8311920, 21.2957935, -67.0969467, 67.0594940
19: -34.4223442, 10.9652939, -34.5006485, 11.0532074, -45.4755516, 45.4659424
20: -30.5165043, 14.2685499, -30.5853405, 14.3551331, -44.8716354, 44.8538895
21: -42.5705643, 14.9051027, -42.6753540, 15.0440941, -57.6146584, 57.5804558
22: -43.0639305, 17.5500488, -43.2472992, 17.7351227, -60.7990532, 60.7973480
23: -34.2730446, 15.1342049, -34.3788300, 15.2624741, -49.5355186, 49.5130348
24: -36.3261261, 14.8802872, -36.3986816, 14.9596519, -51.2857780, 51.2789688
25: -35.4152107, 17.2770710, -35.5269508, 17.4322929, -52.8475037, 52.8040237
26: -53.3623505, 20.1886215, -53.5245285, 20.3205624, -73.6829147, 73.7131500
27: -36.2022400, 18.9068089, -36.2379417, 18.9635658, -55.1658058, 55.1447525
28: -33.1970520, 18.9993744, -33.2920456, 19.0881920, -52.2852440, 52.2914200
29: -44.6859589, 16.7963848, -44.8933029, 17.0142212, -61.7001801, 61.6896896
30: -42.7297134, 20.0112381, -42.8155365, 20.1481743, -62.8778877, 62.8267746
31: -42.2403564, 15.3240032, -42.3121147, 15.4074621, -57.6478195, 57.6361160
32: -38.4656448, 23.1851349, -38.5411987, 23.2029705, -61.6686172, 61.7263336
33: -48.7906876, 35.8787003, -48.9189186, 35.9518661, -84.7425537, 84.7976227
34: -47.1149864, 21.1056595, -47.2015381, 21.1427231, -68.2223663, 68.2696152
35: -41.6633263, 26.4016495, -41.7346840, 26.4311123, -67.6247253, 67.6778259
36: -42.4160614, 26.5970726, -42.5275345, 26.6367970, -68.1737671, 68.2576828
37: -66.8305969, 22.0639820, -66.9691238, 22.2038040, -86.4493332, 86.4846878
38: -52.5285645, 31.0922203, -52.6521835, 31.2026596, -81.8883286, 81.9380722
39: -60.2305641, 35.2730026, -60.3851814, 35.3835678, -95.6141357, 95.6581879
40: -53.4803391, 28.1340084, -53.7004318, 28.3108196, -81.7911606, 81.8344421
41: -39.0469780, 27.0355377, -39.1663361, 27.1299229, -66.1769028, 66.2018738
42: -32.4703674, 21.9787045, -32.5364990, 22.0002575, -54.4706268, 54.5152054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4544908, upper bound: 37.4762770
time: 50.65 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4544908, upper bound: 37.4762770
time: 45.92 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -43.1059608, 35.2412949, -43.1938705, 35.2614365, -78.3674011, 78.4351654
1: -23.3481140, 32.0316048, -23.3925285, 32.0470810, -55.3951950, 55.4241333
2: -18.8386917, 31.9333820, -18.8815098, 31.9474411, -50.7666016, 50.7970619
3: -19.0110588, 35.1693726, -19.0308990, 35.1877937, -53.7841873, 53.7663040
4: -23.5041428, 36.0061264, -23.5605831, 36.0232391, -59.5273819, 59.5664482
5: -21.1920815, 35.5083237, -21.2189598, 35.5325928, -56.0984726, 56.1060905
6: -42.1881104, 26.1218967, -42.2421455, 26.1415672, -68.3296814, 68.3640442
7: -30.4134693, 34.2510147, -30.4534664, 34.2729187, -64.0455704, 64.0704193
8: -29.0175190, 40.1213799, -29.0728149, 40.1486664, -69.1661835, 69.1941986
9: -24.4052219, 31.6620789, -24.4353695, 31.6806412, -54.9441338, 54.9717827
10: -45.8746338, 31.3046074, -45.9105606, 31.3753815, -77.2500153, 77.2151642
11: -48.8961296, 18.2189159, -48.9234238, 18.3129578, -67.2090912, 67.1423416
12: -52.8582077, 18.2169399, -52.8903732, 18.2504864, -69.4631805, 69.4435577
13: -35.7731247, 38.7004471, -35.8134232, 38.7255745, -74.4987030, 74.5138702
14: -78.4816742, 11.0839310, -78.5074081, 11.1757431, -89.6574173, 89.5913391
15: -30.4041023, 30.1361694, -30.4329948, 30.1542130, -60.5583153, 60.5691643
16: -46.2880859, 30.9521103, -46.3379440, 30.9835854, -77.2643204, 77.2846832
17: -77.9502258, 14.7133961, -77.9889603, 14.8477364, -92.7979584, 92.7023544
18: -45.8254128, 21.2778778, -45.8607788, 21.3206348, -67.1460495, 67.1386566
19: -34.5162926, 10.9971666, -34.5395126, 11.0601377, -45.5764313, 45.5366783
20: -30.6108418, 14.3135662, -30.6249943, 14.3663902, -44.9772339, 44.9385605
21: -42.7068214, 14.9623861, -42.7325478, 15.0557384, -57.7625580, 57.6949348
22: -43.3394051, 17.6397781, -43.3676529, 17.7481251, -61.0875320, 61.0074310
23: -34.4241409, 15.1869459, -34.4447174, 15.2736292, -49.6977692, 49.6316643
24: -36.4179306, 14.9169874, -36.4393005, 14.9683657, -51.3862953, 51.3562889
25: -35.5881119, 17.3586082, -35.6032410, 17.4519997, -53.0401115, 52.9618492
26: -53.6026154, 20.2735462, -53.6285973, 20.3334332, -73.9360504, 73.9021454
27: -36.2410240, 18.9459610, -36.2584915, 18.9807892, -55.2218132, 55.2044525
28: -33.3287048, 19.0317802, -33.3504715, 19.1017838, -52.4304886, 52.3822517
29: -44.9962158, 16.8903122, -45.0287361, 17.0255394, -62.0217552, 61.9190483
30: -42.8647881, 20.0729523, -42.8847198, 20.1680374, -63.0328255, 62.9576721
31: -42.3278084, 15.3627443, -42.3490829, 15.4176140, -57.7454224, 57.7118263
32: -38.5265961, 23.2030506, -38.5669861, 23.2155647, -61.7421608, 61.7700348
33: -48.8697548, 35.9808655, -48.9423523, 35.9955597, -84.8653107, 84.9232178
34: -47.1761322, 21.1324005, -47.2252274, 21.1614342, -68.3021393, 68.3193817
35: -41.7126312, 26.4375992, -41.7520065, 26.4461651, -67.6960907, 67.7287140
36: -42.4778061, 26.6376495, -42.5455093, 26.6539192, -68.2698364, 68.3177338
37: -66.8937149, 22.3015308, -66.9988708, 22.3171806, -86.6825485, 86.7629852
38: -52.5839996, 31.2447128, -52.6749420, 31.2745857, -82.0506592, 82.1162262
39: -60.3119812, 35.4540977, -60.4133148, 35.4665909, -95.7785721, 95.8674164
40: -53.5705261, 28.4133625, -53.7198677, 28.4356556, -82.0061798, 82.1332321
41: -39.1141129, 27.1809807, -39.1854095, 27.1942787, -66.3083954, 66.3663940
42: -32.5407257, 22.0134659, -32.5699539, 22.0302391, -54.5709648, 54.5834198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 613

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4544908, upper bound: 37.5172993
time: 75.36 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4544908, upper bound: 37.5172993
time: 53.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 130.94 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.5161551, upper bound: 37.3586012
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.5161551, upper bound: 37.3586012
IS_A1_A1_A2_A1, status: Status.VERIFIED, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.4398975, upper bound: 37.4796551
IS_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.5193923, upper bound: 37.4828745
IS_A1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.4762770, upper bound: 37.4512250
IS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.4762770, upper bound: 37.4951790
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.5172994, upper bound: 37.4512250
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.5172994, upper bound: 37.4951790
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.3777108, upper bound: 37.5161551
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.4573876, upper bound: 37.5193925
IS_A2_B1_B2_B1, status: Status.VERIFIED, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.4540900, upper bound: 37.4398974
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.5021107, upper bound: 37.5193922
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.4544908, upper bound: 37.4762770
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.4544908, upper bound: 37.4762770
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.4544908, upper bound: 37.5172993
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 130.94
Output dim: 8, lower bound: -37.4544908, upper bound: 37.5172993

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -42.8479652, 34.9951897, -43.0053978, 35.1067238, -77.9546890, 78.0005875
1: -23.1989040, 31.8735981, -23.2901707, 31.9484787, -55.1473846, 55.1637688
2: -18.6546822, 31.7503815, -18.7750168, 31.8245831, -50.4538765, 50.5026207
3: -18.8057365, 34.9576035, -18.9520760, 35.0433884, -53.3941727, 53.4688988
4: -23.3218155, 35.8232574, -23.4392567, 35.9287796, -59.2393799, 59.2551804
5: -20.9699726, 35.2273712, -21.1347027, 35.3603096, -55.6802979, 55.7256622
6: -42.0512772, 25.8215408, -42.1375008, 25.9762115, -68.0274887, 67.9590454
7: -30.1614723, 33.9780388, -30.3488140, 34.1163635, -63.6025467, 63.6672974
8: -28.8151779, 39.8771744, -28.9355564, 39.9661713, -68.7813492, 68.8127289
9: -24.1918049, 31.4608231, -24.3285236, 31.6162491, -54.6504555, 54.6222305
10: -45.5108795, 31.0008163, -45.7078743, 31.2434959, -76.7543793, 76.7086945
11: -48.6649933, 17.9815483, -48.7783661, 18.1609802, -66.8259735, 66.7599182
12: -52.4033432, 17.8378162, -52.6320267, 18.1558170, -68.8483887, 68.7315216
13: -35.5373840, 38.5266876, -35.6748695, 38.6279449, -74.1653290, 74.2015533
14: -77.9797897, 10.8088226, -78.2191772, 11.0071125, -88.9869003, 89.0279999
15: -30.0715637, 29.9255352, -30.2601357, 30.1000710, -60.1716347, 60.1856689
16: -46.1268997, 30.7152596, -46.2174110, 30.8607903, -76.9747772, 76.9206161
17: -77.4959946, 14.3342400, -77.7270355, 14.6755714, -92.1715698, 92.0612793
18: -45.6540756, 21.0632343, -45.7044067, 21.1936970, -66.8477707, 66.7676392
19: -34.3259811, 10.8782721, -34.4120483, 10.9631863, -45.2891693, 45.2903214
20: -30.4370613, 14.2043095, -30.5323162, 14.2765141, -44.7135773, 44.7366257
21: -42.4726410, 14.7868023, -42.5841827, 14.9184895, -57.3911285, 57.3709869
22: -42.8763962, 17.3376598, -43.1210480, 17.5895042, -60.4659004, 60.4587097
23: -34.1819115, 15.0166197, -34.3185349, 15.1412725, -49.3231850, 49.3351555
24: -36.1987877, 14.7856121, -36.3187714, 14.8756332, -51.0744209, 51.1043854
25: -35.3205452, 17.1728096, -35.4656601, 17.3179989, -52.6385422, 52.6384697
26: -53.1461067, 19.9051151, -53.3685684, 20.1971512, -73.3432617, 73.2736816
27: -36.0833931, 18.8166771, -36.1797562, 18.8970642, -54.9804573, 54.9964333
28: -33.1057739, 18.8543282, -33.2404785, 18.9754753, -52.0812492, 52.0948067
29: -44.5449028, 16.5763206, -44.7885361, 16.8513031, -61.3962059, 61.3648567
30: -42.5181618, 19.7718506, -42.7458191, 19.9639721, -62.4821320, 62.5176697
31: -42.1698494, 15.2213678, -42.2232170, 15.3160582, -57.4859085, 57.4445839
32: -38.3507690, 23.0514984, -38.4551926, 23.1401806, -61.4909515, 61.5066910
33: -48.6410103, 35.8387337, -48.8152733, 35.9204292, -84.5614395, 84.6540070
34: -47.0031319, 20.9486904, -47.1441956, 21.0542107, -68.0168610, 68.0541611
35: -41.5116081, 26.2719746, -41.6680908, 26.3735428, -67.4073792, 67.4652481
36: -42.3364716, 26.5545349, -42.4340858, 26.6014328, -68.0624847, 68.1235962
37: -66.6669540, 22.0636940, -66.7901306, 22.1853313, -86.2731094, 86.2776642
38: -52.4000015, 31.0035477, -52.5319366, 31.1589508, -81.6850586, 81.6920624
39: -60.0535927, 35.2553596, -60.2292023, 35.3772659, -95.4308624, 95.4845581
40: -53.2813187, 28.0332680, -53.4959946, 28.2085114, -81.4898300, 81.5292664
41: -38.9747047, 26.9496212, -39.0596390, 27.0627823, -66.0374908, 66.0092621
42: -32.4452477, 21.8591385, -32.5147972, 21.9488544, -54.3941040, 54.3739357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 617

## Relational analysis of IS_A1_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5101305, upper bound: 37.3428779
time: 58.03 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5101305, upper bound: 37.3522022
time: 60.16 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -42.8707123, 34.9980202, -43.0570831, 35.1467400, -78.0174561, 78.0550995
1: -23.2157784, 31.8764038, -23.3261948, 31.9784203, -55.1941986, 55.2025986
2: -18.6730881, 31.7531719, -18.8152008, 31.8599720, -50.5084953, 50.5456352
3: -18.8261642, 34.9606361, -18.9951401, 35.0867081, -53.4640808, 53.5152283
4: -23.3336601, 35.8256035, -23.4672928, 35.9490814, -59.2748489, 59.2856598
5: -20.9860229, 35.2305679, -21.1698685, 35.3980904, -55.7373047, 55.7635269
6: -42.0547371, 25.8355942, -42.1650314, 26.0089779, -68.0637131, 68.0006256
7: -30.1760769, 33.9809036, -30.3824673, 34.1474762, -63.6515045, 63.7036591
8: -28.8454323, 39.8821411, -28.9997997, 40.0316620, -68.8770905, 68.8819427
9: -24.1969700, 31.4633980, -24.3395023, 31.6267128, -54.6709976, 54.6374435
10: -45.5179749, 31.0124397, -45.7357521, 31.2729263, -76.7909012, 76.7481918
11: -48.6698837, 17.9948616, -48.8142433, 18.1917076, -66.8615875, 66.8091049
12: -52.4076309, 17.8524189, -52.6698990, 18.1908932, -68.8864822, 68.7927017
13: -35.5601997, 38.5313721, -35.7262421, 38.6736221, -74.2338257, 74.2576141
14: -78.0137253, 10.8121414, -78.2946777, 11.0700150, -89.0837402, 89.1068192
15: -30.0807419, 29.9294052, -30.2864304, 30.1129494, -60.1936913, 60.2158356
16: -46.1317406, 30.7251740, -46.2385941, 30.8850517, -77.0030823, 76.9518814
17: -77.5049820, 14.3385105, -77.7565231, 14.6884060, -92.1933899, 92.0950317
18: -45.6600952, 21.0948448, -45.7658691, 21.2595825, -66.9196777, 66.8607178
19: -34.3304176, 10.8898544, -34.4480629, 10.9871130, -45.3175316, 45.3379173
20: -30.4409790, 14.2136269, -30.5558090, 14.2977791, -44.7387581, 44.7694359
21: -42.4778214, 14.7997217, -42.6207199, 14.9475546, -57.4253769, 57.4204407
22: -42.8824615, 17.3515930, -43.1549416, 17.6191349, -60.5015945, 60.5065346
23: -34.1858902, 15.0224171, -34.3373108, 15.1569376, -49.3428268, 49.3597260
24: -36.2030525, 14.7916832, -36.3484077, 14.8891010, -51.0921555, 51.1400909
25: -35.3249130, 17.1785259, -35.4849930, 17.3346443, -52.6595573, 52.6635208
26: -53.1513863, 19.9265289, -53.4168625, 20.2423763, -73.3937607, 73.3433914
27: -36.0882988, 18.8286152, -36.2123680, 18.9248695, -55.0131683, 55.0409851
28: -33.1094208, 18.8609695, -33.2639465, 18.9925423, -52.1019630, 52.1249161
29: -44.5508461, 16.5881233, -44.8165436, 16.8768559, -61.4277039, 61.4046669
30: -42.5223618, 19.7836609, -42.7697983, 19.9942551, -62.5166168, 62.5534592
31: -42.1753693, 15.2346449, -42.2653732, 15.3442698, -57.5196381, 57.5000191
32: -38.3558960, 23.0617714, -38.4850159, 23.1662312, -61.5221252, 61.5467873
33: -48.6453590, 35.8448486, -48.8380127, 35.9373207, -84.5826797, 84.6828613
34: -47.0069962, 20.9543324, -47.1571121, 21.0695076, -68.0360260, 68.0732574
35: -41.5122108, 26.2759552, -41.6753616, 26.3783207, -67.4129333, 67.4790268
36: -42.3405914, 26.5592499, -42.4524231, 26.6139145, -68.0791397, 68.1502075
37: -66.6718445, 22.0828991, -66.8370209, 22.2282562, -86.3206253, 86.3564606
38: -52.4063416, 31.0067883, -52.5570602, 31.1659813, -81.7103271, 81.7245789
39: -60.0607758, 35.2578430, -60.2546158, 35.3899689, -95.4507446, 95.5124588
40: -53.2869873, 28.0533791, -53.5337563, 28.2543488, -81.5413361, 81.5871353
41: -38.9776688, 26.9658470, -39.0888596, 27.0989723, -66.0766449, 66.0547028
42: -32.4478226, 21.8668060, -32.5255852, 21.9695053, -54.4173279, 54.3923912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 617

## Relational analysis of IS_A1_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5133453, upper bound: 37.4224960
time: 46.61 seconds

## Relational analysis of IS_A1_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5101305, upper bound: 37.3522022
time: 51.84 seconds

## BFS IS instance: IS_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -42.9884949, 35.0679398, -43.0832672, 35.1635361, -78.1520309, 78.1512070
1: -23.2817364, 31.9383926, -23.3379002, 31.9907570, -55.2724915, 55.2762909
2: -18.7801189, 31.8621292, -18.8294296, 31.9048214, -50.6611633, 50.6678772
3: -18.9686928, 35.1165390, -19.0084496, 35.1496086, -53.6768417, 53.6765518
4: -23.4108353, 35.8831596, -23.4889565, 35.9614754, -59.3643036, 59.3721161
5: -21.1454449, 35.4017296, -21.1845169, 35.4628372, -55.9671631, 55.9446716
6: -42.1212349, 25.9616299, -42.1759033, 26.0542431, -68.1754761, 68.1375351
7: -30.3425446, 34.1405334, -30.4018440, 34.2038651, -63.8803177, 63.8781433
8: -28.9434872, 39.9883881, -29.0103416, 40.0668983, -69.0103836, 68.9987335
9: -24.3525047, 31.5945549, -24.4003696, 31.6359749, -54.8300209, 54.8730049
10: -45.7821884, 31.2485123, -45.8489685, 31.2914391, -77.0736237, 77.0974808
11: -48.7369156, 18.0705490, -48.8382645, 18.1881599, -66.9250793, 66.9088135
12: -52.7823143, 18.1710663, -52.8340149, 18.2112274, -69.2681656, 69.2817535
13: -35.7102127, 38.6539459, -35.7718468, 38.6883202, -74.3985291, 74.4257965
14: -78.3491211, 11.0219574, -78.4307709, 11.0766869, -89.4258118, 89.4527283
15: -30.2712364, 30.0904732, -30.3527660, 30.1251087, -60.3963470, 60.4432373
16: -46.1863403, 30.7979164, -46.2691803, 30.8879356, -77.0603180, 77.0545349
17: -77.7393341, 14.5705814, -77.8651581, 14.7011700, -92.4405060, 92.4357376
18: -45.7728500, 21.1945076, -45.8025055, 21.2745056, -67.0473557, 66.9970093
19: -34.3955383, 10.9235687, -34.4680328, 10.9919662, -45.3875046, 45.3916016
20: -30.5322170, 14.2605934, -30.5807133, 14.3062315, -44.8384476, 44.8413086
21: -42.5469437, 14.8540983, -42.6425858, 14.9529953, -57.4999390, 57.4966850
22: -43.0956535, 17.5232964, -43.2310371, 17.6312256, -60.7268791, 60.7543335
23: -34.2661057, 15.0841885, -34.3569183, 15.1760082, -49.4421158, 49.4411087
24: -36.2853851, 14.8472729, -36.3635483, 14.9066067, -51.1919937, 51.2108231
25: -35.3976364, 17.2495117, -35.5050240, 17.3470478, -52.7446823, 52.7545357
26: -53.4867973, 20.1882591, -53.5590401, 20.2645817, -73.7513809, 73.7472992
27: -36.2001038, 18.8926964, -36.2264709, 18.9443436, -55.1444473, 55.1191673
28: -33.2014999, 18.9450855, -33.2787094, 19.0229111, -52.2244110, 52.2237930
29: -44.7188759, 16.7531357, -44.8774796, 16.8863106, -61.6051865, 61.6306152
30: -42.6761742, 19.9630756, -42.7865562, 20.0612450, -62.7374191, 62.7496338
31: -42.2163200, 15.2778759, -42.2844849, 15.3480234, -57.5643425, 57.5623627
32: -38.4703751, 23.1439972, -38.5201263, 23.1816940, -61.6520691, 61.6641235
33: -48.7880554, 35.9317932, -48.8595314, 35.9616737, -84.7497253, 84.7913208
34: -47.1254196, 21.0914993, -47.1711349, 21.1217422, -68.2100372, 68.2238770
35: -41.6405029, 26.3995514, -41.6945839, 26.4287739, -67.6120529, 67.6164474
36: -42.4208832, 26.5974045, -42.4663162, 26.6286926, -68.1881485, 68.1973114
37: -66.7897644, 22.1504860, -66.8647461, 22.2431488, -86.4702148, 86.4510803
38: -52.4969864, 31.0929317, -52.5707436, 31.1942902, -81.8415909, 81.8253632
39: -60.1958084, 35.3466797, -60.2896957, 35.4092941, -95.6051025, 95.6363754
40: -53.4285011, 28.1478291, -53.5529747, 28.2938595, -81.7223587, 81.7008057
41: -39.0401840, 27.0552883, -39.1007957, 27.1286736, -66.1688538, 66.1560822
42: -32.4959793, 21.9642181, -32.5368195, 21.9961319, -54.4921112, 54.5010376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=149, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A1_A1_A2_A2_B1

### Relational analysis result of IS_A1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4757925, upper bound: 37.4791489
time: 54.04 seconds

## Relational analysis of IS_A1_A1_A2_A2_B2

### Relational analysis result of IS_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5168129, upper bound: 37.4791489
time: 52.49 seconds

## BFS IS instance: IS_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -43.0702209, 35.0666695, -42.9967232, 35.0013123, -78.0715332, 78.0633926
1: -23.3189583, 31.9324112, -23.2830563, 31.8939018, -55.2128601, 55.2154694
2: -18.8120956, 31.8423500, -18.7600479, 31.7837982, -50.5758247, 50.5754852
3: -18.9850426, 35.1166840, -18.9715614, 35.0867767, -53.6282959, 53.6254005
4: -23.4525833, 35.8643494, -23.4165115, 35.8031921, -59.2555237, 59.2686119
5: -21.1625481, 35.3789177, -21.1207542, 35.2978439, -55.8323517, 55.8478966
6: -42.1703110, 25.9706230, -42.1320839, 25.9499187, -68.1202316, 68.1027069
7: -30.3711548, 34.1077271, -30.3192101, 34.0367432, -63.7604218, 63.7498779
8: -28.9884415, 39.9967041, -28.9492531, 39.9361115, -68.9245529, 68.9459534
9: -24.3665829, 31.6021976, -24.3435898, 31.5849361, -54.8172607, 54.8127289
10: -45.7623138, 31.3020744, -45.6464043, 31.1830196, -76.9453354, 76.9484787
11: -48.7502403, 18.1507607, -48.7110748, 18.1210365, -66.8712769, 66.8618317
12: -52.7207451, 18.1889439, -52.5882835, 18.1114292, -69.0721893, 69.0955963
13: -35.7205849, 38.6605530, -35.7044296, 38.6269226, -74.3475037, 74.3649826
14: -78.2954712, 11.1066494, -78.1432877, 10.9776421, -89.2731171, 89.2499390
15: -30.2494068, 30.1010227, -30.1904621, 30.0732307, -60.3226395, 60.2914848
16: -46.2292023, 30.7736397, -46.1975327, 30.7220554, -76.9422302, 76.9571533
17: -77.7173233, 14.6966267, -77.5804977, 14.5836525, -92.3009796, 92.2771225
18: -45.7918091, 21.2271385, -45.7858887, 21.2231026, -67.0149078, 67.0130310
19: -34.4208450, 10.9821157, -34.3932800, 10.9603319, -45.3811760, 45.3753967
20: -30.5332966, 14.3090887, -30.4985924, 14.2639694, -44.7972641, 44.8076820
21: -42.5688705, 14.9407377, -42.5321045, 14.8970594, -57.4659309, 57.4728432
22: -43.0859032, 17.6265812, -42.9936600, 17.5439243, -60.6298294, 60.6202393
23: -34.2864342, 15.1633396, -34.2369576, 15.1241913, -49.4106255, 49.4002991
24: -36.3068542, 14.8955402, -36.2903137, 14.8721437, -51.1789970, 51.1858521
25: -35.4056244, 17.3337688, -35.3645630, 17.2697830, -52.6754074, 52.6983337
26: -53.4278679, 20.2456245, -53.3263054, 20.1827469, -73.6106110, 73.5719299
27: -36.2083893, 18.9029312, -36.1917572, 18.8998852, -55.1082764, 55.0946884
28: -33.2165756, 19.0073280, -33.1681709, 18.9923954, -52.2089691, 52.1754990
29: -44.7258263, 16.8797550, -44.6186104, 16.7926273, -61.5184555, 61.4983673
30: -42.7008514, 20.0474701, -42.6857986, 20.0028152, -62.7036667, 62.7332687
31: -42.2368889, 15.3270845, -42.2133827, 15.3105812, -57.5474701, 57.5404663
32: -38.4879532, 23.1451302, -38.4554596, 23.1635532, -61.6515045, 61.6005898
33: -48.8446732, 35.9276466, -48.7864418, 35.8702431, -84.7149200, 84.7140884
34: -47.1592331, 21.1091385, -47.1092415, 21.0976887, -68.2205887, 68.1820374
35: -41.6746826, 26.3942795, -41.6502991, 26.3939571, -67.6129532, 67.5705414
36: -42.4755630, 26.6104431, -42.4071045, 26.5937214, -68.2134399, 68.1360550
37: -66.8809204, 22.1428890, -66.8099365, 22.0465565, -86.3954010, 86.3512268
38: -52.5727997, 31.1110916, -52.5203400, 31.0691071, -81.8288269, 81.7661285
39: -60.2805252, 35.3362350, -60.2127609, 35.2541275, -95.5346527, 95.5489960
40: -53.5712090, 28.1713009, -53.4681435, 28.0760593, -81.6472702, 81.6394424
41: -39.1054840, 27.0491219, -39.0390549, 27.0038490, -66.1093292, 66.0881805
42: -32.4989929, 21.9632568, -32.4653091, 21.9660416, -54.4650345, 54.4285660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 984

## Relational analysis of IS_A1_A2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3962636, upper bound: 37.4915189
time: 49.80 seconds

## Relational analysis of IS_A1_A2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4757925, upper bound: 37.4946937
time: 88.25 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -42.9757614, 35.0817261, -43.0678825, 35.1772766, -78.1530380, 78.1496124
1: -23.2673378, 31.9195538, -23.3296661, 31.9917526, -55.2590904, 55.2492218
2: -18.7233372, 31.7913666, -18.8184891, 31.8702526, -50.5732498, 50.5881233
3: -18.8557720, 34.9882050, -18.9968758, 35.0907440, -53.4977722, 53.5546532
4: -23.4003696, 35.8848419, -23.4714279, 35.9641953, -59.3645630, 59.3528252
5: -21.0189266, 35.2848358, -21.1720963, 35.4116096, -55.7976608, 55.8250618
6: -42.1176834, 25.8988857, -42.1697845, 26.0303764, -68.1480560, 68.0686722
7: -30.2236824, 34.0240974, -30.3864632, 34.1562424, -63.7284622, 63.7590942
8: -28.9033070, 39.9560394, -29.0012970, 40.0544357, -68.9577408, 68.9573364
9: -24.2322502, 31.4956703, -24.3367195, 31.6329842, -54.7208290, 54.6481895
10: -45.5925369, 31.0914955, -45.7543259, 31.2757988, -76.8683319, 76.8458252
11: -48.7508698, 18.0898247, -48.8361282, 18.1933289, -66.9441986, 66.9259491
12: -52.4538498, 17.8892460, -52.6755753, 18.1914139, -68.9528351, 68.8848038
13: -35.5928192, 38.5655823, -35.7233658, 38.6763382, -74.2691574, 74.2889481
14: -78.0940018, 10.9104729, -78.3208160, 11.0725460, -89.1665497, 89.2312927
15: -30.1312141, 29.9543762, -30.2958870, 30.1156940, -60.2469101, 60.2502632
16: -46.1971512, 30.7742558, -46.2455864, 30.8913860, -77.0807190, 77.0092773
17: -77.6180191, 14.4804344, -77.7973557, 14.6917744, -92.3097916, 92.2777863
18: -45.7083092, 21.1509476, -45.7738190, 21.2569008, -66.9652100, 66.9247665
19: -34.3942413, 10.9545908, -34.4675598, 10.9877224, -45.3819656, 45.4221497
20: -30.4812870, 14.2724018, -30.5684128, 14.3015461, -44.7828331, 44.8408127
21: -42.5564804, 14.8971720, -42.6468430, 14.9492836, -57.5057640, 57.5440140
22: -42.9927635, 17.4657898, -43.1936378, 17.6236534, -60.6164169, 60.6594276
23: -34.2715225, 15.1123829, -34.3688889, 15.1581335, -49.4296570, 49.4812698
24: -36.2642784, 14.8478956, -36.3677788, 14.8921738, -51.1564522, 51.2156754
25: -35.4087372, 17.2819519, -35.5177536, 17.3395386, -52.7482758, 52.7997055
26: -53.1960640, 19.9926491, -53.4248886, 20.2493496, -73.4454117, 73.4175415
27: -36.1166000, 18.8557930, -36.2169685, 18.9199791, -55.0365791, 55.0727615
28: -33.1826057, 18.9357605, -33.2853355, 18.9952641, -52.1778717, 52.2210960
29: -44.6925087, 16.7250443, -44.8686676, 16.8781509, -61.5706596, 61.5937119
30: -42.6157188, 19.8867569, -42.8044395, 19.9983673, -62.6140862, 62.6911964
31: -42.2321167, 15.2930298, -42.2823067, 15.3461256, -57.5782433, 57.5753365
32: -38.3987389, 23.0751534, -38.4817657, 23.1663399, -61.5650787, 61.5569191
33: -48.7248688, 35.8838654, -48.8445358, 35.9485931, -84.6734619, 84.7284012
34: -47.0639725, 20.9899178, -47.1568985, 21.0726795, -68.0966263, 68.1107101
35: -41.5599518, 26.2853355, -41.6840019, 26.3797760, -67.4635773, 67.5051422
36: -42.4123459, 26.5889740, -42.4554176, 26.6195240, -68.1662445, 68.1828995
37: -66.7919693, 22.1880817, -66.8459091, 22.2697315, -86.5302124, 86.4833374
38: -52.5020676, 31.0963898, -52.5647430, 31.1923180, -81.8686676, 81.8343658
39: -60.1729355, 35.3299942, -60.2596970, 35.4162521, -95.5891876, 95.5896912
40: -53.4486923, 28.2010212, -53.5395660, 28.3166828, -81.7653732, 81.7405853
41: -39.0612793, 27.0233059, -39.0950470, 27.1200829, -66.1813660, 66.1183548
42: -32.4833870, 21.8954582, -32.5251083, 21.9749146, -54.4583015, 54.4205666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 613

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 984

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5136355, upper bound: 37.3711185
time: 57.52 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5168132, upper bound: 37.4507427
time: 54.30 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -43.0963020, 35.1519051, -43.0913048, 35.1938553, -78.2901611, 78.2432098
1: -23.3341618, 31.9817810, -23.3405952, 32.0038185, -55.3379822, 55.3223763
2: -18.8312321, 31.9004898, -18.8318787, 31.9149399, -50.7270355, 50.7092934
3: -18.9991913, 35.1445656, -19.0093079, 35.1532211, -53.7140579, 53.7124863
4: -23.4778671, 35.9428482, -23.4927979, 35.9761162, -59.4539833, 59.4356461
5: -21.1792564, 35.4560928, -21.1858768, 35.4762421, -56.0298462, 56.0038757
6: -42.1851540, 26.0254498, -42.1796799, 26.0750732, -68.2602234, 68.2051315
7: -30.3915749, 34.1838379, -30.4044895, 34.2123680, -63.9602890, 63.9305954
8: -29.0025082, 40.0628052, -29.0107002, 40.0890617, -69.0915680, 69.0735016
9: -24.3882332, 31.6276436, -24.3971214, 31.6414852, -54.8816223, 54.8819656
10: -45.8572388, 31.3284683, -45.8669472, 31.2934875, -77.1507263, 77.1954193
11: -48.8185158, 18.1668282, -48.8596420, 18.1890507, -67.0075684, 67.0264740
12: -52.8289795, 18.2087784, -52.8391304, 18.2109318, -69.3291626, 69.3791199
13: -35.7438965, 38.6886940, -35.7678337, 38.6904755, -74.4343719, 74.4565277
14: -78.4309311, 11.1204796, -78.4553528, 11.0790176, -89.5099487, 89.5758362
15: -30.3227463, 30.1161118, -30.3610973, 30.1272430, -60.4499893, 60.4772110
16: -46.2526398, 30.8475552, -46.2753754, 30.8937244, -77.1374359, 77.1125412
17: -77.8567886, 14.7131767, -77.9013519, 14.7038765, -92.5606689, 92.6145325
18: -45.8214722, 21.2518806, -45.8101463, 21.2705002, -67.0919724, 67.0620270
19: -34.4596558, 10.9888535, -34.4871368, 10.9920511, -45.4517059, 45.4759903
20: -30.5729294, 14.3203583, -30.5929108, 14.3090305, -44.8819580, 44.9132690
21: -42.6260109, 14.9521503, -42.6682892, 14.9541492, -57.5801620, 57.6204376
22: -43.2063599, 17.6396141, -43.2691727, 17.6336384, -60.8399963, 60.9087868
23: -34.3523026, 15.1744232, -34.3879662, 15.1769257, -49.5292282, 49.5623894
24: -36.3474731, 14.9043274, -36.3820724, 14.9088316, -51.2563057, 51.2863998
25: -35.4818382, 17.3535309, -35.5373993, 17.3513451, -52.8331833, 52.8909302
26: -53.5319366, 20.2583904, -53.5665512, 20.2676086, -73.7995453, 73.8249435
27: -36.2289047, 18.9203281, -36.2305832, 18.9389038, -55.1678085, 55.1509094
28: -33.2750244, 19.0209084, -33.2997742, 19.0248775, -52.2999039, 52.3206825
29: -44.8612175, 16.8910789, -44.9288254, 16.8865871, -61.7478027, 61.8199043
30: -42.7700195, 20.0673866, -42.8207703, 20.0646305, -62.8346481, 62.8881569
31: -42.2737656, 15.3370533, -42.3007050, 15.3491783, -57.6229439, 57.6377563
32: -38.5138245, 23.1577511, -38.5162811, 23.1814384, -61.6952629, 61.6740341
33: -48.8679886, 35.9713173, -48.8657150, 35.9723663, -84.8403549, 84.8370361
34: -47.1829567, 21.1277466, -47.1703415, 21.1243477, -68.2703552, 68.2616806
35: -41.6919136, 26.4093628, -41.6996193, 26.4298820, -67.6636429, 67.6418457
36: -42.4934006, 26.6275024, -42.4688110, 26.6339474, -68.2735519, 68.2318420
37: -66.9105911, 22.2562351, -66.8730164, 22.2840042, -86.6735229, 86.5843277
38: -52.5954666, 31.1828156, -52.5758133, 31.2202606, -82.0068054, 81.9284973
39: -60.3085556, 35.4192505, -60.2942238, 35.4352303, -95.7437897, 95.7134705
40: -53.5906677, 28.2960892, -53.5583382, 28.3553562, -81.9460220, 81.8544312
41: -39.1245575, 27.1133595, -39.1062012, 27.1491890, -66.2737427, 66.2195587
42: -32.5323868, 21.9933681, -32.5355301, 22.0010338, -54.5334206, 54.5289001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 613

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 984

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5136355, upper bound: 37.4152479
time: 57.45 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5136355, upper bound: 37.4946934
time: 53.04 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -43.0158234, 35.1427269, -42.9352036, 35.0794830, -78.0953064, 78.0779266
1: -23.2952557, 31.9691334, -23.2519627, 31.9253006, -55.2205582, 55.2210960
2: -18.7793961, 31.8403244, -18.6996880, 31.7917747, -50.5498047, 50.5158539
3: -18.9513416, 35.0492363, -18.8321991, 34.9804192, -53.5041046, 53.4300385
4: -23.4437256, 35.9561653, -23.3898621, 35.8978882, -59.3333130, 59.3292732
5: -21.1382523, 35.3813286, -21.0040741, 35.2849274, -55.7924957, 55.7406731
6: -42.1432571, 26.0056438, -42.1024971, 25.9016991, -68.0449524, 68.1081390
7: -30.3543739, 34.1399307, -30.2160988, 34.0392036, -63.7420044, 63.6899567
8: -28.9390945, 39.9951096, -28.8782406, 39.9556656, -68.8947601, 68.8733521
9: -24.3325691, 31.6343918, -24.2308712, 31.5086956, -54.6920128, 54.7231216
10: -45.7099228, 31.2518406, -45.5524445, 31.0413456, -76.7512665, 76.8042831
11: -48.8086243, 18.1642303, -48.7591820, 18.0715942, -66.8802185, 66.9234161
12: -52.6443329, 18.1589966, -52.4509125, 17.8724899, -68.8011093, 68.9181747
13: -35.6709061, 38.6344147, -35.5907097, 38.5559731, -74.2268829, 74.2251282
14: -78.2376251, 11.0104446, -78.0408020, 10.8600578, -89.0976868, 89.0512466
15: -30.2768669, 30.1063385, -30.1336861, 29.9579830, -60.2348480, 60.2400246
16: -46.2262917, 30.8900490, -46.2046700, 30.7943382, -77.0112534, 77.0840836
17: -77.7658157, 14.6819801, -77.6074600, 14.4620037, -92.2278214, 92.2894440
18: -45.7096939, 21.1964836, -45.6681442, 21.1236820, -66.8333740, 66.8646240
19: -34.4384804, 10.9659901, -34.4004211, 10.9447222, -45.3832016, 45.3664093
20: -30.5474243, 14.2791786, -30.4836559, 14.2458744, -44.7932968, 44.7628326
21: -42.6196442, 14.9210196, -42.5715065, 14.8784552, -57.4981003, 57.4925270
22: -43.1690063, 17.5929852, -43.0029755, 17.4408226, -60.6098289, 60.5959625
23: -34.3526459, 15.1437950, -34.2688789, 15.0998211, -49.4524689, 49.4126740
24: -36.3516006, 14.8777800, -36.2843323, 14.8386803, -51.1902809, 51.1621132
25: -35.5078888, 17.3225136, -35.4244423, 17.2654648, -52.7733536, 52.7469559
26: -53.3918877, 20.2004395, -53.2167740, 19.9748936, -73.3667831, 73.4172134
27: -36.1874466, 18.8933563, -36.1071892, 18.8579769, -55.0454254, 55.0005455
28: -33.2674026, 18.9781036, -33.1766815, 18.9265079, -52.1939087, 52.1547852
29: -44.8514862, 16.8526840, -44.7036476, 16.7061615, -61.5576477, 61.5563316
30: -42.7854462, 19.9671650, -42.6228218, 19.8614731, -62.6469193, 62.5899887
31: -42.2473793, 15.3196020, -42.2393494, 15.2810183, -57.5283966, 57.5589523
32: -38.4621353, 23.1587772, -38.3971825, 23.1029949, -61.5651321, 61.5559616
33: -48.8152580, 35.9254494, -48.7068405, 35.8563004, -84.6715546, 84.6322937
34: -47.1476212, 21.0587254, -47.0404968, 20.9759903, -68.0850143, 68.0584564
35: -41.6779976, 26.3777351, -41.5652771, 26.2992554, -67.5046310, 67.4694214
36: -42.4384766, 26.6027889, -42.3764191, 26.5766563, -68.1524658, 68.1035156
37: -66.8010864, 22.1966972, -66.7309341, 22.1148376, -86.3592529, 86.3359528
38: -52.5366096, 31.1749744, -52.4672546, 31.0813313, -81.7918701, 81.7765961
39: -60.2362289, 35.3935738, -60.1359940, 35.2977066, -95.5339355, 95.5295715
40: -53.5047150, 28.2610874, -53.4025383, 28.1603203, -81.6650391, 81.6636276
41: -39.0652618, 27.0849800, -39.0302315, 27.0108795, -66.0761414, 66.1152115
42: -32.5168152, 21.9541817, -32.4772987, 21.8805733, -54.3973885, 54.4314804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 617

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3613628, upper bound: 37.5101305
time: 46.98 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3719101, upper bound: 37.5101305
time: 56.08 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -43.0674858, 35.1827316, -42.9579582, 35.0823250, -78.1498108, 78.1406860
1: -23.3313065, 31.9991150, -23.2688255, 31.9280930, -55.2593994, 55.2679405
2: -18.8195992, 31.8757133, -18.7180862, 31.7945671, -50.5928116, 50.5704613
3: -18.9943867, 35.0925980, -18.8526459, 34.9834595, -53.5503845, 53.4999008
4: -23.4717712, 35.9764595, -23.4016838, 35.9002495, -59.3638229, 59.3647003
5: -21.1734238, 35.4191055, -21.0200882, 35.2881241, -55.8303833, 55.7976418
6: -42.1708107, 26.0384216, -42.1059265, 25.9157200, -68.0865326, 68.1443481
7: -30.3880463, 34.1710587, -30.2307110, 34.0421295, -63.7783737, 63.7388840
8: -29.0033360, 40.0606003, -28.9085178, 39.9606514, -68.9639893, 68.9691162
9: -24.3435593, 31.6448650, -24.2360439, 31.5112705, -54.7072144, 54.7436485
10: -45.7377739, 31.2812653, -45.5595474, 31.0529671, -76.7907410, 76.8408127
11: -48.8445053, 18.1949158, -48.7640686, 18.0848885, -66.9293976, 66.9589844
12: -52.6821632, 18.1941357, -52.4552078, 17.8871002, -68.8622665, 68.9563293
13: -35.7222748, 38.6801262, -35.6135292, 38.5606384, -74.2829132, 74.2936554
14: -78.3130951, 11.0733757, -78.0747604, 10.8633919, -89.1764832, 89.1481323
15: -30.3031502, 30.1191769, -30.1428661, 29.9618568, -60.2650070, 60.2620430
16: -46.2475014, 30.9142799, -46.2095184, 30.8042526, -77.0425491, 77.1124268
17: -77.7952652, 14.6948051, -77.6164474, 14.4662685, -92.2615356, 92.3112488
18: -45.7711372, 21.2623692, -45.6741638, 21.1552925, -66.9264297, 66.9365311
19: -34.4744720, 10.9899235, -34.4048615, 10.9563160, -45.4307861, 45.3947830
20: -30.5709114, 14.3004313, -30.4875870, 14.2552109, -44.8261223, 44.7880173
21: -42.6561584, 14.9500904, -42.5766602, 14.8913698, -57.5475273, 57.5267487
22: -43.2029457, 17.6226120, -43.0090370, 17.4547653, -60.6577110, 60.6316490
23: -34.3714066, 15.1594610, -34.2728691, 15.1056376, -49.4770432, 49.4323311
24: -36.3812408, 14.8912449, -36.2886276, 14.8447552, -51.2259979, 51.1798706
25: -35.5272255, 17.3391495, -35.4287949, 17.2711754, -52.7984009, 52.7679443
26: -53.4401970, 20.2456512, -53.2220154, 19.9963112, -73.4365082, 73.4676666
27: -36.2200699, 18.9211464, -36.1120834, 18.8699379, -55.0900078, 55.0332298
28: -33.2908669, 18.9951630, -33.1803246, 18.9331322, -52.2239990, 52.1754875
29: -44.8794899, 16.8782387, -44.7095718, 16.7179794, -61.5974693, 61.5878105
30: -42.8094330, 19.9974651, -42.6269951, 19.8732834, -62.6827164, 62.6244583
31: -42.2895508, 15.3478270, -42.2448845, 15.2942686, -57.5838203, 57.5927124
32: -38.4919510, 23.1848221, -38.4022827, 23.1132679, -61.6052170, 61.5871048
33: -48.8379517, 35.9424133, -48.7112122, 35.8624344, -84.7003860, 84.6536255
34: -47.1605797, 21.0740395, -47.0443420, 20.9816113, -68.1041641, 68.0775757
35: -41.6852875, 26.3825760, -41.5658417, 26.3032341, -67.5184479, 67.4750061
36: -42.4568329, 26.6152668, -42.3805313, 26.5813904, -68.1790466, 68.1201630
37: -66.8479538, 22.2396240, -66.7358322, 22.1340218, -86.4380417, 86.3835068
38: -52.5617447, 31.1820259, -52.4736328, 31.0845737, -81.8244781, 81.8019104
39: -60.2616501, 35.4062691, -60.1431427, 35.3001900, -95.5618439, 95.5494080
40: -53.5425110, 28.3068829, -53.4082069, 28.1804237, -81.7229309, 81.7150879
41: -39.0944901, 27.1211834, -39.0331955, 27.0271168, -66.1216049, 66.1543808
42: -32.5275955, 21.9748249, -32.4798737, 21.8882294, -54.4158249, 54.4546967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 617

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4409151, upper bound: 37.5133452
time: 55.68 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3719101, upper bound: 37.5101305
time: 176.45 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -43.0936546, 35.1995468, -43.0757523, 35.1522751, -78.2459259, 78.2752991
1: -23.3430176, 32.0114212, -23.3348656, 31.9901218, -55.3331375, 55.3462868
2: -18.8338051, 31.9205875, -18.8251705, 31.9035149, -50.7150497, 50.7231560
3: -19.0076904, 35.1554947, -18.9951668, 35.1393967, -53.7117462, 53.7126770
4: -23.4934788, 35.9888382, -23.4788933, 35.9577980, -59.4503708, 59.4542618
5: -21.1880741, 35.4838715, -21.1795578, 35.4592819, -56.0115128, 56.0275536
6: -42.1816826, 26.0836945, -42.1724091, 26.0417061, -68.2233887, 68.2561035
7: -30.4073925, 34.2274666, -30.3972626, 34.2016068, -63.9527435, 63.9678040
8: -29.0138626, 40.0958443, -29.0065899, 40.0668564, -69.0807190, 69.1024323
9: -24.4044380, 31.6541061, -24.3915577, 31.6424370, -54.9427719, 54.9026756
10: -45.8510094, 31.2997780, -45.8236732, 31.2890930, -77.1401062, 77.1234512
11: -48.8685226, 18.1913872, -48.8311539, 18.1607780, -67.0292969, 67.0225372
12: -52.8462830, 18.2144489, -52.8298416, 18.2057438, -69.3514023, 69.3379211
13: -35.7678719, 38.6948013, -35.7634697, 38.6832352, -74.4511108, 74.4582672
14: -78.4492493, 11.0800171, -78.4101181, 11.0732079, -89.5224609, 89.4901352
15: -30.3695107, 30.1313477, -30.3332500, 30.1229477, -60.4924583, 60.4645996
16: -46.2781029, 30.9171696, -46.2641525, 30.8769836, -77.1451874, 77.1697235
17: -77.9039078, 14.7075768, -77.8506622, 14.6983681, -92.6022797, 92.5582428
18: -45.8078079, 21.2772598, -45.7870178, 21.2549706, -67.0627747, 67.0642776
19: -34.4944496, 10.9947586, -34.4699478, 10.9900360, -45.4844856, 45.4647064
20: -30.5958233, 14.3089037, -30.5788288, 14.3021717, -44.8979950, 44.8877335
21: -42.6780357, 14.9555779, -42.6457558, 14.9457207, -57.6237564, 57.6013336
22: -43.2790222, 17.6347084, -43.2221069, 17.6264820, -60.9055023, 60.8568153
23: -34.3910217, 15.1785498, -34.3530998, 15.1673899, -49.5584106, 49.5316505
24: -36.3963547, 14.9087458, -36.3709869, 14.9003630, -51.2967186, 51.2797318
25: -35.5472183, 17.3515396, -35.5015717, 17.3421593, -52.8893776, 52.8531113
26: -53.5823822, 20.2678776, -53.5573959, 20.2580433, -73.8404236, 73.8252716
27: -36.2341385, 18.9406281, -36.2238922, 18.9339924, -55.1681290, 55.1645203
28: -33.3056488, 19.0255318, -33.2723961, 19.0172405, -52.3228912, 52.2979279
29: -44.9404221, 16.8877201, -44.8774872, 16.8829536, -61.8233757, 61.7652054
30: -42.8262024, 20.0644493, -42.7808609, 20.0528946, -62.8790970, 62.8453102
31: -42.3086624, 15.3515854, -42.2858353, 15.3375607, -57.6462250, 57.6374207
32: -38.5270653, 23.2002525, -38.5167160, 23.1954670, -61.7225342, 61.7169685
33: -48.8595428, 35.9667587, -48.8539085, 35.9493370, -84.8088837, 84.8206635
34: -47.1745605, 21.1263199, -47.1628189, 21.1186752, -68.2547302, 68.2516479
35: -41.7045021, 26.4329948, -41.6941452, 26.4268456, -67.6558838, 67.6741257
36: -42.4707108, 26.6300201, -42.4611702, 26.6195927, -68.2261658, 68.2294464
37: -66.8756409, 22.2545719, -66.8538055, 22.2016068, -86.5326233, 86.5332031
38: -52.5754318, 31.2103233, -52.5642471, 31.1707077, -81.9252167, 81.9335022
39: -60.2967300, 35.4255753, -60.2782364, 35.3890419, -95.6857758, 95.7038116
40: -53.5617371, 28.3463955, -53.5496750, 28.2746410, -81.8363800, 81.8960724
41: -39.1063919, 27.1508904, -39.0956879, 27.1165657, -66.2229614, 66.2465820
42: -32.5388184, 22.0014610, -32.5280418, 21.9856491, -54.5244675, 54.5295029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=150, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 617

## Relational analysis of IS_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4856113, upper bound: 37.5133449
time: 56.39 seconds

## Relational analysis of IS_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4961027, upper bound: 37.5133449
time: 56.11 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -43.0783844, 35.2134094, -43.0651932, 35.1679153, -78.2462997, 78.2786026
1: -23.3348236, 32.0125275, -23.3209839, 31.9713249, -55.3061485, 55.3335114
2: -18.8229904, 31.8860359, -18.7690048, 31.8328037, -50.6354523, 50.6359596
3: -18.9961739, 35.0967102, -18.8826237, 35.0110550, -53.5885925, 53.5340729
4: -23.4761848, 35.9920540, -23.4695683, 35.9608879, -59.4325867, 59.4563675
5: -21.1758308, 35.4327011, -21.0536919, 35.3425140, -55.8921967, 55.8587494
6: -42.1756859, 26.0599308, -42.1697311, 25.9793587, -68.1550446, 68.2296600
7: -30.3921661, 34.1798592, -30.2790909, 34.0853119, -63.8340302, 63.8167191
8: -29.0048866, 40.0840683, -28.9671860, 40.0364876, -69.0413742, 69.0512543
9: -24.3407841, 31.6512947, -24.2714424, 31.5440636, -54.7184639, 54.7940521
10: -45.7566757, 31.2842369, -45.6349602, 31.1331215, -76.8898010, 76.9191971
11: -48.8682632, 18.1966915, -48.8471565, 18.1827202, -67.0509796, 67.0438461
12: -52.6879425, 18.1947098, -52.5014458, 17.9254513, -68.9559784, 69.0233917
13: -35.7193909, 38.6829834, -35.6462555, 38.5958061, -74.3152008, 74.3292389
14: -78.3394318, 11.0759678, -78.1552277, 10.9626656, -89.3020935, 89.2311935
15: -30.3126736, 30.1220779, -30.1938381, 29.9874878, -60.3001633, 60.3159180
16: -46.2544861, 30.9206390, -46.2751007, 30.8533554, -77.0999298, 77.1902542
17: -77.8361816, 14.6982803, -77.7298355, 14.6088982, -92.4450836, 92.4281158
18: -45.7794418, 21.2617245, -45.7284431, 21.2146473, -66.9940872, 66.9901657
19: -34.4940872, 10.9905243, -34.4690514, 11.0212021, -45.5152893, 45.4595757
20: -30.5835819, 14.3041935, -30.5280170, 14.3146706, -44.8982544, 44.8322105
21: -42.6824265, 14.9518509, -42.6571884, 14.9891281, -57.6715546, 57.6090393
22: -43.2416916, 17.6271915, -43.1195107, 17.5693054, -60.8109970, 60.7467041
23: -34.4030190, 15.1607370, -34.3598862, 15.1963196, -49.5993385, 49.5206223
24: -36.4006615, 14.8943291, -36.3500366, 14.9010811, -51.3017426, 51.2443657
25: -35.5600281, 17.3442955, -35.5127411, 17.3756104, -52.9356384, 52.8570366
26: -53.4482574, 20.2525997, -53.2669106, 20.0625019, -73.5107574, 73.5195084
27: -36.2247925, 18.9162979, -36.1409073, 18.8973083, -55.1221008, 55.0572052
28: -33.3123093, 18.9979897, -33.2542572, 19.0084419, -52.3207512, 52.2522469
29: -44.9316254, 16.8797684, -44.8513336, 16.8554554, -61.7870789, 61.7311020
30: -42.8445816, 20.0016861, -42.7226639, 19.9772758, -62.8218575, 62.7243500
31: -42.3066559, 15.3496990, -42.3019028, 15.3529758, -57.6596298, 57.6516037
32: -38.4889030, 23.1849537, -38.4456787, 23.1268883, -61.6157913, 61.6306305
33: -48.8445244, 35.9537506, -48.7911682, 35.9016571, -84.7461853, 84.7449188
34: -47.1604271, 21.0771942, -47.1016350, 21.0169125, -68.1415863, 68.1385880
35: -41.6939697, 26.3840733, -41.6139259, 26.3153229, -67.5464630, 67.5264053
36: -42.4599152, 26.6215439, -42.4550705, 26.6118202, -68.2121582, 68.2083359
37: -66.8570023, 22.2848244, -66.8605194, 22.2443352, -86.5651855, 86.5942688
38: -52.5697021, 31.2132721, -52.5747871, 31.1811562, -81.9417114, 81.9683533
39: -60.2668839, 35.4325333, -60.2560005, 35.3723640, -95.6392517, 95.6885376
40: -53.5484123, 28.3692684, -53.5715942, 28.3300991, -81.8785095, 81.9408646
41: -39.1007423, 27.1423759, -39.1177979, 27.0847797, -66.1855240, 66.2601776
42: -32.5279922, 21.9804535, -32.5164185, 21.9183426, -54.4463348, 54.4968719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 613

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 984

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3745319, upper bound: 37.5136354
time: 51.01 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4728359, upper bound: 37.5168131
time: 53.48 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -43.1018066, 35.2299881, -43.1857491, 35.2381248, -78.3399353, 78.4157410
1: -23.3457680, 32.0245819, -23.3878727, 32.0335922, -55.3793602, 55.4124527
2: -18.8363724, 31.9307327, -18.8769398, 31.9419136, -50.7566605, 50.7897339
3: -19.0086079, 35.1591949, -19.0260525, 35.1674309, -53.7463989, 53.7503967
4: -23.4975681, 36.0039597, -23.5470867, 36.0188942, -59.5164642, 59.5474434
5: -21.1895905, 35.4973183, -21.2140274, 35.5137367, -56.0710068, 56.0909424
6: -42.1855545, 26.1046658, -42.2371864, 26.1058426, -68.2913971, 68.3418503
7: -30.4102364, 34.2360001, -30.4470577, 34.2449951, -64.0054169, 64.0485840
8: -29.0142727, 40.1186867, -29.0663738, 40.1432037, -69.1574783, 69.1850586
9: -24.4011955, 31.6597939, -24.4274197, 31.6760750, -54.9522209, 54.9548454
10: -45.8692894, 31.3018990, -45.8996277, 31.3700657, -77.2393570, 77.2015228
11: -48.8917618, 18.1924286, -48.9149094, 18.2599220, -67.1516876, 67.1073380
12: -52.8515015, 18.2141972, -52.8765984, 18.2450428, -69.4503174, 69.3997498
13: -35.7638702, 38.6971436, -35.7973022, 38.7189636, -74.4828339, 74.4944458
14: -78.4739609, 11.0824471, -78.4921722, 11.1726418, -89.6466064, 89.5746155
15: -30.3778858, 30.1336174, -30.3852711, 30.1492329, -60.5271187, 60.5188904
16: -46.2843208, 30.9230003, -46.3306885, 30.9266357, -77.2031708, 77.2470551
17: -77.9402313, 14.7103539, -77.9684830, 14.8416367, -92.7818680, 92.6788330
18: -45.8158188, 21.2753296, -45.8417206, 21.3156471, -67.1314697, 67.1170502
19: -34.5136795, 10.9948339, -34.5344543, 11.0554657, -45.5691452, 45.5292892
20: -30.6080933, 14.3116922, -30.6196785, 14.3626385, -44.9707336, 44.9313698
21: -42.7038498, 14.9567108, -42.7266998, 15.0441093, -57.7479591, 57.6834106
22: -43.3172379, 17.6371822, -43.3329697, 17.7430992, -61.0603371, 60.9701538
23: -34.4220886, 15.1795149, -34.4407120, 15.2583761, -49.6804657, 49.6202278
24: -36.4149513, 14.9110088, -36.4332886, 14.9575224, -51.3724747, 51.3442993
25: -35.5796661, 17.3561249, -35.5858650, 17.4471893, -53.0268555, 52.9419899
26: -53.5899582, 20.2708664, -53.6026917, 20.3282642, -73.9182205, 73.8735580
27: -36.2383881, 18.9352341, -36.2532501, 18.9618492, -55.2002373, 55.1884842
28: -33.3267555, 19.0276108, -33.3467064, 19.0936317, -52.4203873, 52.3743172
29: -44.9918137, 16.8882027, -45.0199165, 17.0213966, -62.0132103, 61.9081192
30: -42.8609314, 20.0679245, -42.8770485, 20.1580734, -63.0190048, 62.9449730
31: -42.3250008, 15.3527641, -42.3435936, 15.3970547, -57.7220535, 57.6963577
32: -38.5233994, 23.2000160, -38.5606766, 23.2094955, -61.7328949, 61.7606926
33: -48.8657265, 35.9775200, -48.9342918, 35.9891129, -84.8548431, 84.9118118
34: -47.1738815, 21.1289215, -47.2206345, 21.1546535, -68.2925720, 68.3124084
35: -41.7095833, 26.4341946, -41.7458954, 26.4393311, -67.6831665, 67.7264481
36: -42.4732742, 26.6359081, -42.5362740, 26.6504326, -68.2611389, 68.3159027
37: -66.8840637, 22.2990875, -66.9792175, 22.3125324, -86.6662292, 86.7377167
38: -52.5807266, 31.2412357, -52.6684074, 31.2676582, -82.0358734, 82.1066284
39: -60.3013992, 35.4515228, -60.3916130, 35.4616318, -95.7630310, 95.8431396
40: -53.5672150, 28.4079781, -53.7134781, 28.4250183, -81.9922333, 82.1214600
41: -39.1118698, 27.1715012, -39.1810837, 27.1748238, -66.2866974, 66.3525848
42: -32.5384369, 22.0065727, -32.5654488, 22.0162430, -54.5546799, 54.5720215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 613

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 984

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3745319, upper bound: 37.5136354
time: 46.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5168128, upper bound: 37.5168131
time: 52.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 101.76 seconds
IS_A1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.5101305, upper bound: 37.3428779
IS_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.5101305, upper bound: 37.3522022
IS_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.5133453, upper bound: 37.4224960
IS_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.5101305, upper bound: 37.3522022
IS_A1_A1_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.4757925, upper bound: 37.4791489
IS_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.5168129, upper bound: 37.4791489
IS_A1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.3962636, upper bound: 37.4915189
IS_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.4757925, upper bound: 37.4946937
IS_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.5136355, upper bound: 37.3711185
IS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.5168132, upper bound: 37.4507427
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.5136355, upper bound: 37.4152479
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.5136355, upper bound: 37.4946934
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.3613628, upper bound: 37.5101305
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.3719101, upper bound: 37.5101305
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.4409151, upper bound: 37.5133452
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.3719101, upper bound: 37.5101305
IS_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.4856113, upper bound: 37.5133449
IS_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.4961027, upper bound: 37.5133449
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.3745319, upper bound: 37.5136354
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.4728359, upper bound: 37.5168131
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.3745319, upper bound: 37.5136354
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 101.76
Output dim: 8, lower bound: -37.5168128, upper bound: 37.5168131

## BFS IS instance: IS_A1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -42.7522812, 34.8884239, -42.9874115, 35.0548592, -77.8071442, 77.8758392
1: -23.1472187, 31.8227062, -23.2814178, 31.9255829, -55.0727997, 55.1041260
2: -18.6216297, 31.7140541, -18.7680454, 31.8081779, -50.4016113, 50.4590912
3: -18.7657509, 34.9305611, -18.9380703, 35.0330124, -53.3367004, 53.4245567
4: -23.2614880, 35.7445259, -23.4300613, 35.8925629, -59.1396561, 59.1706161
5: -20.9462147, 35.1737328, -21.1282711, 35.3387871, -55.6267319, 55.6643524
6: -42.0111313, 25.7361679, -42.1288605, 25.9390106, -67.9501419, 67.8650284
7: -30.1100025, 33.9206886, -30.3386536, 34.0911942, -63.5120010, 63.5981979
8: -28.7690277, 39.8086700, -28.9282722, 39.9379997, -68.7070312, 68.7369385
9: -24.1349392, 31.3782692, -24.3198338, 31.5783005, -54.5240250, 54.5301552
10: -45.4616318, 30.9739380, -45.6932297, 31.2297668, -76.6913986, 76.6671677
11: -48.5440063, 17.9034958, -48.7264671, 18.1546345, -66.6986389, 66.6299591
12: -52.3524055, 17.8046875, -52.6143913, 18.1492119, -68.7821350, 68.6494751
13: -35.4785995, 38.4840012, -35.6592865, 38.6144180, -74.0930176, 74.1432877
14: -77.9000702, 10.7615547, -78.1883011, 11.0009727, -88.9010468, 88.9498596
15: -29.9912891, 29.8715935, -30.2294350, 30.0871601, -60.0784492, 60.1010284
16: -46.0289307, 30.5884819, -46.2008667, 30.8027210, -76.8161316, 76.7762833
17: -77.3412247, 14.1898079, -77.6616516, 14.6627865, -92.0040131, 91.8514557
18: -45.6346130, 20.9801865, -45.6909485, 21.1600704, -66.7946854, 66.6711349
19: -34.2335205, 10.8195324, -34.3738594, 10.9587450, -45.1922646, 45.1933899
20: -30.3688641, 14.1634693, -30.5045223, 14.2716818, -44.6405449, 44.6679916
21: -42.3378563, 14.7016039, -42.5288658, 14.9151592, -57.2530136, 57.2304688
22: -42.7015610, 17.2293625, -43.0424423, 17.5821896, -60.2837524, 60.2718048
23: -34.0844994, 14.9517336, -34.2759705, 15.1371517, -49.2216492, 49.2277031
24: -36.0759811, 14.7349529, -36.2642860, 14.8722916, -50.9482727, 50.9992371
25: -35.1695633, 17.0774708, -35.3967209, 17.3097572, -52.4793205, 52.4741898
26: -53.0617256, 19.8457451, -53.3349533, 20.1916580, -73.2533875, 73.1806946
27: -36.0527420, 18.7578316, -36.1645508, 18.8788414, -54.9315834, 54.9223824
28: -33.0342865, 18.7996807, -33.2094994, 18.9707394, -52.0050278, 52.0091782
29: -44.3423233, 16.4490662, -44.6958160, 16.8435211, -61.1858444, 61.1448822
30: -42.3732147, 19.6820068, -42.6808319, 19.9575043, -62.3307190, 62.3628387
31: -42.0668182, 15.1557512, -42.1804962, 15.3106756, -57.3774948, 57.3362465
32: -38.2900658, 22.9676666, -38.4407768, 23.1024990, -61.3925629, 61.4084435
33: -48.5436249, 35.8154068, -48.7845459, 35.9086266, -84.4522552, 84.5999527
34: -46.9530258, 20.8765907, -47.1369247, 21.0259666, -67.9380493, 67.9740829
35: -41.4694939, 26.2464447, -41.6561432, 26.3632812, -67.3475494, 67.4163818
36: -42.2797394, 26.4851589, -42.4260178, 26.5722370, -67.9540405, 68.0380096
37: -66.5823669, 21.9747009, -66.7687378, 22.1410847, -86.1205521, 86.1551514
38: -52.3188934, 30.8603306, -52.5181656, 31.0880470, -81.4812851, 81.5102081
39: -59.9553642, 35.1793365, -60.2128525, 35.3418694, -95.2972336, 95.3921890
40: -53.1573792, 27.8702450, -53.4786606, 28.1290970, -81.2864761, 81.3489075
41: -38.9015884, 26.8457336, -39.0504684, 27.0138092, -65.9153976, 65.8962021
42: -32.4014511, 21.8362312, -32.5003815, 21.9391727, -54.3406219, 54.3366127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=149, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A1_A1_A1_B1_A1_A1

### Relational analysis result of IS_A1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5076208, upper bound: 37.3060459
time: 52.13 seconds

## Relational analysis of IS_A1_A1_A1_B1_A1_A2

### Relational analysis result of IS_A1_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5076208, upper bound: 37.3398001
time: 49.30 seconds

## BFS IS instance: IS_A1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -42.8447304, 34.9901352, -43.0038300, 35.1044960, -77.9492264, 77.9939651
1: -23.1980534, 31.8707733, -23.2897377, 31.9471378, -55.1451912, 55.1605110
2: -18.6538029, 31.7475548, -18.7746048, 31.8232155, -50.4533386, 50.4982605
3: -18.7969856, 34.9555206, -18.9480400, 35.0423660, -53.3895035, 53.4616776
4: -23.3198280, 35.8186493, -23.4383335, 35.9266205, -59.2410278, 59.2453270
5: -20.9677601, 35.2244263, -21.1336555, 35.3588600, -55.6827621, 55.7190208
6: -42.0490608, 25.8111000, -42.1364441, 25.9711704, -68.0202332, 67.9475403
7: -30.1602745, 33.9741058, -30.3482704, 34.1145287, -63.6084747, 63.6579132
8: -28.8140678, 39.8726997, -28.9350548, 39.9641724, -68.7782440, 68.8077545
9: -24.1887989, 31.4579964, -24.3271141, 31.6148834, -54.6707535, 54.6085205
10: -45.4959335, 30.9984093, -45.7007523, 31.2423649, -76.7382965, 76.6991577
11: -48.6596222, 17.9799500, -48.7758026, 18.1602345, -66.8198547, 66.7557526
12: -52.3989334, 17.8362122, -52.6298447, 18.1550579, -68.8319702, 68.7424622
13: -35.5263176, 38.5229721, -35.6697845, 38.6261597, -74.1524811, 74.1927567
14: -77.9753494, 10.8069286, -78.2171021, 11.0062046, -88.9815521, 89.0240326
15: -30.0663853, 29.9237747, -30.2577038, 30.0992165, -60.1656036, 60.1814804
16: -46.1239548, 30.7105732, -46.2159958, 30.8583260, -76.9713593, 76.9130936
17: -77.4905853, 14.3324947, -77.7245331, 14.6747437, -92.1653290, 92.0570297
18: -45.6504707, 21.0369797, -45.7027054, 21.1822205, -66.8326874, 66.7396851
19: -34.3214645, 10.8777933, -34.4098930, 10.9629660, -45.2844315, 45.2876854
20: -30.4334869, 14.2028217, -30.5304508, 14.2758007, -44.7092896, 44.7332726
21: -42.4664497, 14.7861805, -42.5810471, 14.9181833, -57.3846321, 57.3672256
22: -42.8696136, 17.3366203, -43.1178665, 17.5890293, -60.4586411, 60.4544868
23: -34.1767349, 15.0155106, -34.3157501, 15.1407528, -49.3174896, 49.3312607
24: -36.1940079, 14.7841759, -36.3164482, 14.8749752, -51.0689850, 51.1006241
25: -35.3154373, 17.1706142, -35.4625244, 17.3169937, -52.6324310, 52.6331406
26: -53.1417427, 19.9044704, -53.3664856, 20.1968517, -73.3385925, 73.2709579
27: -36.0798149, 18.7972679, -36.1780396, 18.8881397, -54.9679565, 54.9753075
28: -33.1021843, 18.8533268, -33.2386703, 18.9750023, -52.0771866, 52.0919952
29: -44.5367813, 16.5734901, -44.7843666, 16.8499985, -61.3867798, 61.3578568
30: -42.5131226, 19.7699814, -42.7433701, 19.9631157, -62.4762383, 62.5133514
31: -42.1654701, 15.2207603, -42.2210922, 15.3157549, -57.4812241, 57.4418526
32: -38.3470573, 23.0469437, -38.4534454, 23.1380424, -61.4850998, 61.5003891
33: -48.6223831, 35.8349915, -48.8058853, 35.9186325, -84.5410156, 84.6408768
34: -46.9999466, 20.9420776, -47.1427193, 21.0510578, -68.0105743, 68.0439224
35: -41.5079117, 26.2700768, -41.6663208, 26.3726234, -67.3973846, 67.4617310
36: -42.3340378, 26.5389977, -42.4329605, 26.5930195, -68.0573120, 68.0919724
37: -66.6640549, 22.0541725, -66.7887573, 22.1808586, -86.2675171, 86.2123108
38: -52.3957710, 30.9911461, -52.5299339, 31.1532974, -81.6773224, 81.6357880
39: -60.0510178, 35.2523346, -60.2279282, 35.3758392, -95.4268570, 95.4802628
40: -53.2780876, 28.0231152, -53.4945068, 28.2045631, -81.4826508, 81.5176239
41: -38.9721985, 26.9441185, -39.0584335, 27.0597572, -66.0319519, 66.0025482
42: -32.4371796, 21.8569469, -32.5110397, 21.9478378, -54.3850174, 54.3679886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=149, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1642

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4667646, upper bound: 37.3494258
time: 66.36 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4667646, upper bound: 37.3494258
time: 55.16 seconds

## BFS IS instance: IS_A1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -42.7750244, 34.8912125, -43.0390816, 35.0948448, -77.8698730, 77.9302979
1: -23.1640625, 31.8254833, -23.3174515, 31.9555664, -55.1196289, 55.1429367
2: -18.6400337, 31.7168236, -18.8082275, 31.8435402, -50.4562302, 50.5021019
3: -18.7862282, 34.9335976, -18.9811478, 35.0763779, -53.4065781, 53.4708443
4: -23.2733383, 35.7469177, -23.4580994, 35.9128761, -59.1751328, 59.2011070
5: -20.9622421, 35.1769028, -21.1634445, 35.3765945, -55.6837158, 55.7022667
6: -42.0145912, 25.7502079, -42.1564140, 25.9718151, -67.9864044, 67.9066238
7: -30.1245956, 33.9235535, -30.3723297, 34.1223030, -63.5609894, 63.6345673
8: -28.7992897, 39.8136330, -28.9924984, 40.0035095, -68.8027954, 68.8061295
9: -24.1401081, 31.3808346, -24.3308258, 31.5888062, -54.5445137, 54.5453682
10: -45.4687538, 30.9855595, -45.7210617, 31.2591858, -76.7279358, 76.7066193
11: -48.5489044, 17.9167976, -48.7623749, 18.1853142, -66.7342224, 66.6791687
12: -52.3566742, 17.8192749, -52.6522751, 18.1843262, -68.8202591, 68.7106781
13: -35.5014114, 38.4886475, -35.7106857, 38.6601105, -74.1615219, 74.1993332
14: -77.9339905, 10.7648544, -78.2637634, 11.0638981, -88.9978867, 89.0286179
15: -30.0004616, 29.8754654, -30.2557316, 30.1000443, -60.1005058, 60.1311951
16: -46.0337524, 30.5984116, -46.2220764, 30.8269539, -76.8444519, 76.8075333
17: -77.3501968, 14.1940880, -77.6911087, 14.6756134, -92.0258102, 91.8851929
18: -45.6406326, 21.0118332, -45.7524643, 21.2259502, -66.8665848, 66.7642975
19: -34.2379608, 10.8311253, -34.4098511, 10.9826918, -45.2206535, 45.2409744
20: -30.3727474, 14.1727686, -30.5279999, 14.2929649, -44.6657104, 44.7007675
21: -42.3429985, 14.7145386, -42.5653839, 14.9442129, -57.2872124, 57.2799225
22: -42.7076492, 17.2432880, -43.0763512, 17.6118450, -60.3194962, 60.3196411
23: -34.0884705, 14.9575205, -34.2947540, 15.1528206, -49.2412910, 49.2522736
24: -36.0802383, 14.7410240, -36.2939339, 14.8857679, -50.9660072, 51.0349579
25: -35.1739349, 17.0831985, -35.4160690, 17.3264179, -52.5003510, 52.4992676
26: -53.0669708, 19.8671665, -53.3832397, 20.2368355, -73.3038025, 73.2504044
27: -36.0576401, 18.7697678, -36.1971779, 18.9066391, -54.9642792, 54.9669456
28: -33.0379372, 18.8063087, -33.2329559, 18.9878082, -52.0257454, 52.0392647
29: -44.3482780, 16.4609051, -44.7238426, 16.8690414, -61.2173195, 61.1847458
30: -42.3774185, 19.6938515, -42.7048264, 19.9877892, -62.3652077, 62.3986778
31: -42.0723267, 15.1689930, -42.2226448, 15.3388739, -57.4112015, 57.3916397
32: -38.2951851, 22.9779358, -38.4705811, 23.1285458, -61.4237289, 61.4485168
33: -48.5479813, 35.8214874, -48.8072357, 35.9255981, -84.4735794, 84.6287231
34: -46.9569092, 20.8822060, -47.1498489, 21.0412254, -67.9572296, 67.9932022
35: -41.4700241, 26.2503948, -41.6634293, 26.3680878, -67.3531036, 67.4301529
36: -42.2838287, 26.4898872, -42.4443817, 26.5846539, -67.9707718, 68.0646362
37: -66.5873260, 21.9938889, -66.8156128, 22.1840172, -86.1681213, 86.2339172
38: -52.3252258, 30.8635826, -52.5432701, 31.0950699, -81.5065613, 81.5428009
39: -59.9625397, 35.1818123, -60.2382507, 35.3545532, -95.3170929, 95.4200592
40: -53.1630707, 27.8903847, -53.5164833, 28.1748981, -81.3379669, 81.4068680
41: -38.9045486, 26.8619576, -39.0797195, 27.0499916, -65.9545441, 65.9416809
42: -32.4040070, 21.8438911, -32.5111580, 21.9598351, -54.3638420, 54.3550491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=149, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A1_A1_A1_B2_A1_A1

### Relational analysis result of IS_A1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5107861, upper bound: 37.3856652
time: 49.19 seconds

## Relational analysis of IS_A1_A1_A1_B2_A1_A2

### Relational analysis result of IS_A1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5107861, upper bound: 37.4193296
time: 77.58 seconds

## BFS IS instance: IS_A1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -42.8674736, 34.9929581, -43.0555077, 35.1445007, -78.0119781, 78.0484619
1: -23.2148991, 31.8735695, -23.3257790, 31.9770985, -55.1919975, 55.1993484
2: -18.6722126, 31.7503681, -18.8148003, 31.8586044, -50.5079880, 50.5412674
3: -18.8174477, 34.9585419, -18.9911041, 35.0857201, -53.4593964, 53.5079918
4: -23.3316574, 35.8210068, -23.4663658, 35.9469147, -59.2764816, 59.2758369
5: -20.9837780, 35.2276154, -21.1688538, 35.3966560, -55.7397537, 55.7568855
6: -42.0525284, 25.8251190, -42.1639786, 26.0039520, -68.0564804, 67.9890976
7: -30.1748638, 33.9769821, -30.3818741, 34.1456566, -63.6574478, 63.6942902
8: -28.8443470, 39.8776855, -28.9992943, 40.0296402, -68.8739853, 68.8769836
9: -24.1939602, 31.4605865, -24.3381062, 31.6253777, -54.6912918, 54.6237335
10: -45.5030518, 31.0100441, -45.7285995, 31.2717781, -76.7748260, 76.7386475
11: -48.6645126, 17.9932289, -48.8117027, 18.1909218, -66.8554382, 66.8049316
12: -52.4032478, 17.8508263, -52.6677132, 18.1901360, -68.8700485, 68.8036804
13: -35.5491638, 38.5276260, -35.7211761, 38.6718369, -74.2210007, 74.2488022
14: -78.0092773, 10.8102036, -78.2925720, 11.0691223, -89.0783997, 89.1027756
15: -30.0755520, 29.9276676, -30.2840271, 30.1120720, -60.1876221, 60.2116928
16: -46.1287689, 30.7205162, -46.2371979, 30.8825607, -76.9996948, 76.9443817
17: -77.4995575, 14.3367691, -77.7539444, 14.6876240, -92.1871796, 92.0907135
18: -45.6564636, 21.0685921, -45.7641449, 21.2481117, -66.9045715, 66.8327332
19: -34.3259125, 10.8893785, -34.4459076, 10.9868937, -45.3128052, 45.3352852
20: -30.4373837, 14.2121410, -30.5539284, 14.2970772, -44.7344589, 44.7660675
21: -42.4716034, 14.7990885, -42.6175690, 14.9472694, -57.4188728, 57.4166565
22: -42.8756943, 17.3505650, -43.1517715, 17.6186523, -60.4943466, 60.5023346
23: -34.1807175, 15.0213137, -34.3345070, 15.1564045, -49.3371201, 49.3558197
24: -36.1982841, 14.7902584, -36.3460922, 14.8884430, -51.0867271, 51.1363525
25: -35.3198204, 17.1763306, -35.4818649, 17.3336220, -52.6534424, 52.6581955
26: -53.1469803, 19.9258823, -53.4147797, 20.2420368, -73.3890152, 73.3406601
27: -36.0847015, 18.8092461, -36.2106781, 18.9159412, -55.0006409, 55.0199242
28: -33.1058350, 18.8599644, -33.2621384, 18.9920673, -52.0979004, 52.1221008
29: -44.5427170, 16.5853119, -44.8123894, 16.8755264, -61.4182434, 61.3977013
30: -42.5172920, 19.7817745, -42.7673264, 19.9934006, -62.5106926, 62.5491028
31: -42.1709976, 15.2340183, -42.2632675, 15.3439560, -57.5149536, 57.4972839
32: -38.3521576, 23.0571899, -38.4832764, 23.1641121, -61.5162697, 61.5404663
33: -48.6267433, 35.8411140, -48.8285599, 35.9355888, -84.5623322, 84.6696777
34: -47.0037956, 20.9477005, -47.1556320, 21.0663338, -68.0297699, 68.0630341
35: -41.5084534, 26.2740440, -41.6735840, 26.3774376, -67.4029541, 67.4755173
36: -42.3381500, 26.5437317, -42.4513054, 26.6054821, -68.0738983, 68.1186218
37: -66.6690216, 22.0733929, -66.8356018, 22.2238274, -86.3150330, 86.2910614
38: -52.4021378, 30.9943962, -52.5550461, 31.1603413, -81.7026367, 81.6683655
39: -60.0581741, 35.2548256, -60.2533302, 35.3885269, -95.4467010, 95.5081558
40: -53.2837448, 28.0432415, -53.5322723, 28.2503815, -81.5341263, 81.5755157
41: -38.9751320, 26.9603672, -39.0876541, 27.0959530, -66.0710831, 66.0480194
42: -32.4397507, 21.8646126, -32.5218163, 21.9684868, -54.4082375, 54.3864288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=149, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1016
type: B, layer: 1, pos: 1016
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1015
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1015
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1014
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1014
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1642

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A1_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4699184, upper bound: 37.4288769
time: 54.90 seconds

## Relational analysis of IS_A1_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5107861, upper bound: 37.4288769
time: 58.30 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 115.57 seconds
IS_A1_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 115.57
Output dim: 8, lower bound: -37.5076208, upper bound: 37.3060459
IS_A1_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 115.57
Output dim: 8, lower bound: -37.5076208, upper bound: 37.3398001
IS_A1_A1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 115.57
Output dim: 8, lower bound: -37.4667646, upper bound: 37.3494258
IS_A1_A1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 115.57
Output dim: 8, lower bound: -37.4667646, upper bound: 37.3494258
IS_A1_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 115.57
Output dim: 8, lower bound: -37.5107861, upper bound: 37.3856652
IS_A1_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 115.57
Output dim: 8, lower bound: -37.5107861, upper bound: 37.4193296
IS_A1_A1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 115.57
Output dim: 8, lower bound: -37.4699184, upper bound: 37.4288769
IS_A1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 115.57
Output dim: 8, lower bound: -37.5107861, upper bound: 37.4288769
IS_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.5168129, upper bound: 37.4791489
IS_A1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.3962636, upper bound: 37.4915189
IS_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.4757925, upper bound: 37.4946937
IS_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.5136355, upper bound: 37.3711185
IS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.5168132, upper bound: 37.4507427
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.5136355, upper bound: 37.4152479
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.5136355, upper bound: 37.4946934
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.3613628, upper bound: 37.5101305
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.3719101, upper bound: 37.5101305
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.4409151, upper bound: 37.5133452
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.3719101, upper bound: 37.5101305
IS_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.4856113, upper bound: 37.5133449
IS_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.4961027, upper bound: 37.5133449
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.3745319, upper bound: 37.5136354
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.4728359, upper bound: 37.5168131
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.3745319, upper bound: 37.5136354
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 115.57
Output dim: 8, lower bound: -37.5168128, upper bound: 37.5168131

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 80.06 + 3612.25 = 3692.31 seconds
