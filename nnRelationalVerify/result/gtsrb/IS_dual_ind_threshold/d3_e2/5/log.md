## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 5)
Time budget: 7200 seconds
Split limit: 100
Threshold: 38.9746791072


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328)
1: (-31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664)
2: (-30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690)
3: (-34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942)
4: (-40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316)
5: (-37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755)
6: (-56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726)
7: (-43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831)
8: (-39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809)
9: (-34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275)
10: (-55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739)
11: (-56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778)
12: (-59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474)
13: (-48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827)
14: (-81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586)
15: (-40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336)
16: (-58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498)
17: (-85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696)
18: (-49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788)
19: (-41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662)
20: (-35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226)
21: (-49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371)
22: (-51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297)
23: (-39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429)
24: (-45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464)
25: (-38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314)
26: (-59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385)
27: (-49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097)
28: (-37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770)
29: (-55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838)
30: (-47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052)
31: (-49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653)
32: (-49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205)
33: (-72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273)
34: (-61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420)
35: (-57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084)
36: (-57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705)
37: (-85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175)
38: (-69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412)
39: (-85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074)
40: (-75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985)
41: (-54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703)
42: (-39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.89 + 100.90 = 103.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -39.0136928, upper bound: 39.0136928

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 664

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0024557, upper bound: 38.8860657
time: 80.99 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0024557, upper bound: 38.8860657
time: 103.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 184.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 184.69
Output dim: 2, lower bound: -39.0024557, upper bound: 38.8860657
IS_A2, status: Status.UNKNOWN, split count: 1, time: 184.69
Output dim: 2, lower bound: -39.0024557, upper bound: 38.8860657

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -53.3774796, 42.9201660, -53.4858742, 43.0329552, -96.4104309, 96.4060287
1: -31.6713161, 35.9829025, -31.7502480, 36.0713043, -67.7426147, 67.7331467
2: -30.4362183, 35.4602814, -30.5226498, 35.6113129, -66.0475311, 65.9829254
3: -33.9908981, 41.4721298, -34.0679703, 41.6151924, -75.6060867, 75.5401001
4: -40.0867233, 38.7826843, -40.1650238, 38.9286003, -79.0153198, 78.9477081
5: -36.9322891, 41.2082977, -37.0118484, 41.3846626, -78.3169403, 78.2201385
6: -55.9646416, 22.4740601, -55.9983978, 22.5356407, -78.5002747, 78.4724503
7: -42.9315414, 39.9677620, -43.0661774, 40.1735382, -83.1050797, 83.0339355
8: -39.4270706, 45.3632355, -39.5130920, 45.5292358, -84.9562988, 84.8763199
9: -34.1458015, 37.4384384, -34.2567749, 37.5418053, -71.6876068, 71.6952057
10: -55.0700035, 52.3056755, -55.2491035, 52.4441032, -107.5141068, 107.5547791
11: -56.5478859, 39.7582512, -56.5856133, 39.8061218, -96.3540039, 96.3438644
12: -58.7168579, 43.9561768, -59.0824585, 44.1457329, -102.8625870, 103.0386353
13: -48.6515274, 49.5336685, -48.8072662, 49.6778259, -98.3293533, 98.3409271
14: -81.2405090, 43.3291359, -81.5581131, 43.4678230, -124.7083282, 124.8872528
15: -40.3422050, 36.3591537, -40.4699020, 36.4281311, -76.7703323, 76.8290482
16: -58.2661934, 40.8014603, -58.3909798, 40.8968887, -99.1630783, 99.1924438
17: -84.9234009, 62.4777832, -85.2327118, 62.6251755, -147.5485687, 147.7104950
18: -48.9506607, 29.1751289, -49.0728378, 29.2251987, -78.1758575, 78.2479706
19: -41.3661690, 19.5413876, -41.4458771, 19.5716972, -60.9378662, 60.9872627
20: -35.3639450, 21.8191853, -35.4567642, 21.8672924, -57.2312317, 57.2759476
21: -49.1743317, 25.4946327, -49.2634048, 25.5329361, -74.7072601, 74.7580414
22: -50.8104973, 30.0631008, -51.0258942, 30.1738853, -80.9843826, 81.0889893
23: -39.1500854, 26.6619015, -39.2320175, 26.6859913, -65.8360748, 65.8939209
24: -45.2300682, 22.8305588, -45.3285332, 22.9052315, -68.1352997, 68.1590881
25: -38.5401802, 31.0744190, -38.6140900, 31.1292629, -69.6694412, 69.6885071
26: -58.7275848, 37.6064453, -59.0705910, 37.7739487, -96.5015335, 96.6770325
27: -49.3559227, 27.3421707, -49.4714622, 27.4109631, -76.7668839, 76.8136292
28: -37.8543472, 28.8824539, -37.9314041, 28.9160995, -66.7704468, 66.8138580
29: -55.3245239, 34.3495483, -55.5191917, 34.4601326, -89.7846527, 89.8687439
30: -47.8210258, 27.2703438, -47.8846893, 27.3164501, -75.1374741, 75.1550293
31: -48.9871292, 23.9611149, -49.1138725, 24.0706406, -73.0577698, 73.0749817
32: -49.0948563, 27.4573383, -49.2001724, 27.5326042, -76.6274567, 76.6575089
33: -71.8713837, 43.9789734, -71.9776993, 44.1067352, -115.9781189, 115.9566650
34: -60.9188461, 30.0975037, -60.9967461, 30.1329899, -91.0518265, 91.0942535
35: -57.2233429, 34.7466774, -57.3010254, 34.7860031, -92.0093384, 92.0476990
36: -57.1787148, 33.9547424, -57.3194237, 34.0418320, -91.2205505, 91.2741623
37: -85.2632599, 33.1793442, -85.3698654, 33.2305794, -118.4938354, 118.5492096
38: -69.0542297, 40.9801598, -69.1707077, 41.0557404, -110.1099701, 110.1508636
39: -85.0638275, 40.7166672, -85.1463165, 40.8183022, -125.8821259, 125.8629837
40: -75.1952820, 29.8964272, -75.2944641, 30.0250168, -105.2202911, 105.1908875
41: -54.3153839, 26.0054893, -54.3868446, 26.0599384, -80.3753204, 80.3923187
42: -38.8637657, 29.4154549, -38.9626999, 29.4932079, -68.3569717, 68.3781586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=406, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 632

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8924653, upper bound: 38.8829812
time: 72.85 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8924653, upper bound: 38.8847501
time: 141.73 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -53.5135803, 43.0761452, -53.5196304, 43.0867310, -96.6003036, 96.5957642
1: -31.7654037, 36.1058731, -31.7692986, 36.1147881, -67.8801880, 67.8751678
2: -30.5365067, 35.6818161, -30.5394077, 35.6860580, -66.2225647, 66.2212219
3: -34.0809555, 41.6812744, -34.0839233, 41.6860619, -75.7670135, 75.7651978
4: -40.1791306, 38.9950562, -40.1818314, 39.0005417, -79.1796722, 79.1768875
5: -37.0262070, 41.4658279, -37.0287933, 41.4710617, -78.4972610, 78.4946213
6: -56.0252609, 22.5471497, -56.0296288, 22.5597305, -78.5849915, 78.5767822
7: -43.0875397, 40.2722473, -43.0929947, 40.2788696, -83.3664093, 83.3652420
8: -39.5256615, 45.6052246, -39.5296173, 45.6103897, -85.1360474, 85.1348343
9: -34.3050423, 37.5688248, -34.3087158, 37.5728722, -71.8779144, 71.8775330
10: -55.3274994, 52.4715958, -55.3338814, 52.4757690, -107.8032684, 107.8054733
11: -56.6291237, 39.8087654, -56.6361198, 39.8227654, -96.4518890, 96.4448853
12: -59.2548561, 44.1688919, -59.2646980, 44.1721077, -103.4269638, 103.4335861
13: -48.8757477, 49.7082977, -48.8808556, 49.7140846, -98.5898285, 98.5891571
14: -81.6982574, 43.4821167, -81.7085114, 43.4844017, -125.1826630, 125.1906204
15: -40.5029068, 36.4573441, -40.5265923, 36.4625549, -76.9654617, 76.9839325
16: -58.4302902, 40.9145126, -58.4360275, 40.9427567, -99.3730469, 99.3505402
17: -85.3743515, 62.6414108, -85.3829193, 62.6454277, -148.0197754, 148.0243225
18: -49.1168289, 29.2405167, -49.1220398, 29.2439919, -78.3608246, 78.3625565
19: -41.4786072, 19.5760727, -41.4817009, 19.5862103, -61.0648117, 61.0577736
20: -35.4935303, 21.8775063, -35.4974976, 21.8791771, -57.3727074, 57.3750038
21: -49.3049660, 25.5434704, -49.3100510, 25.5470486, -74.8520126, 74.8535156
22: -51.1147652, 30.1895676, -51.1281738, 30.1948299, -81.3095856, 81.3177414
23: -39.2672501, 26.6934509, -39.2705078, 26.6969604, -65.9642105, 65.9639587
24: -45.3528099, 22.9371681, -45.3572083, 22.9400387, -68.2928467, 68.2943726
25: -38.6375427, 31.1472244, -38.6469498, 31.1509380, -69.7884827, 69.7941742
26: -59.2280502, 37.7941360, -59.2382965, 37.7993317, -97.0273743, 97.0324249
27: -49.5023918, 27.4402447, -49.5081444, 27.4445457, -76.9469223, 76.9483871
28: -37.9619064, 28.9277020, -37.9643402, 28.9300346, -66.8919373, 66.8920441
29: -55.6044312, 34.4713745, -55.6125526, 34.4747696, -90.0792007, 90.0839233
30: -47.9092979, 27.3305931, -47.9146652, 27.3325462, -75.2418365, 75.2452545
31: -49.1493149, 24.1174660, -49.1538544, 24.1214714, -73.2707825, 73.2713165
32: -49.2460747, 27.5472507, -49.2515297, 27.5497589, -76.7958298, 76.7987823
33: -72.0010681, 44.1612091, -72.0042572, 44.1665039, -116.1675720, 116.1654663
34: -61.0207672, 30.1560001, -61.0235863, 30.1594925, -91.1802521, 91.1795807
35: -57.3228912, 34.8069344, -57.3258743, 34.8100052, -92.1328964, 92.1328049
36: -57.3833542, 34.0570068, -57.3889313, 34.0601692, -91.4435196, 91.4459305
37: -85.4078064, 33.2505569, -85.4125214, 33.2538872, -118.6616974, 118.6630707
38: -69.2103882, 41.0769730, -69.2196045, 41.0799484, -110.2903366, 110.2965698
39: -85.1711884, 40.8650131, -85.1750870, 40.8693619, -126.0405502, 126.0401001
40: -75.3145752, 30.0851860, -75.3199615, 30.0893326, -105.4039001, 105.4051514
41: -54.4125786, 26.0732460, -54.4161034, 26.0831833, -80.4957581, 80.4893494
42: -39.0074234, 29.5120354, -39.0121994, 29.5147972, -68.5222168, 68.5242310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 632

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8924653, upper bound: 38.8829812
time: 76.10 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0103459, upper bound: 39.0103455
time: 80.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 159.04 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 159.04
Output dim: 2, lower bound: -38.8924653, upper bound: 38.8829812
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 159.04
Output dim: 2, lower bound: -38.8924653, upper bound: 38.8847501
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 159.04
Output dim: 2, lower bound: -38.8924653, upper bound: 38.8829812
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 159.04
Output dim: 2, lower bound: -39.0103459, upper bound: 39.0103455

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -53.5106926, 43.0735779, -53.5096474, 43.0778389, -96.5885315, 96.5832214
1: -31.7634640, 36.1041412, -31.7626457, 36.1087112, -67.8721771, 67.8667908
2: -30.5352516, 35.6797523, -30.5351505, 35.6787872, -66.2140350, 66.2149048
3: -34.0797310, 41.6784477, -34.0798492, 41.6762009, -75.7559357, 75.7583008
4: -40.1769371, 38.9919090, -40.1743126, 38.9896088, -79.1665497, 79.1662140
5: -37.0250015, 41.4630241, -37.0246544, 41.4612045, -78.4862061, 78.4876785
6: -56.0225677, 22.5406570, -56.0205269, 22.5369797, -78.5595474, 78.5611877
7: -43.0850754, 40.2693748, -43.0845604, 40.2688179, -83.3538895, 83.3539276
8: -39.5233917, 45.6021690, -39.5218811, 45.5997543, -85.1231461, 85.1240540
9: -34.2985573, 37.5662079, -34.2872849, 37.5639725, -71.8625336, 71.8534851
10: -55.3240509, 52.4685326, -55.3218536, 52.4651680, -107.7892151, 107.7903824
11: -56.6248055, 39.8067245, -56.6213379, 39.8157234, -96.4405212, 96.4280624
12: -59.2504768, 44.1671791, -59.2494583, 44.1661911, -103.4166718, 103.4166336
13: -48.8689346, 49.7040634, -48.8569145, 49.6994247, -98.5683594, 98.5609741
14: -81.6926880, 43.4806290, -81.6890869, 43.4792023, -125.1718903, 125.1697083
15: -40.4935532, 36.4543152, -40.4960480, 36.4521294, -76.9456787, 76.9503632
16: -58.4271240, 40.9081306, -58.4250984, 40.9226151, -99.3497391, 99.3332291
17: -85.3686829, 62.6395264, -85.3630676, 62.6389084, -148.0075684, 148.0025940
18: -49.1118698, 29.2278023, -49.1049194, 29.2010479, -78.3129120, 78.3327179
19: -41.4764328, 19.5749359, -41.4741287, 19.5822830, -61.0587158, 61.0490608
20: -35.4912453, 21.8762417, -35.4896431, 21.8747406, -57.3659859, 57.3658829
21: -49.3018646, 25.5420628, -49.2992973, 25.5420895, -74.8439560, 74.8413620
22: -51.1108208, 30.1872902, -51.1144028, 30.1870251, -81.2978363, 81.3016891
23: -39.2648277, 26.6920834, -39.2620735, 26.6922493, -65.9570770, 65.9541550
24: -45.3498306, 22.9354172, -45.3470116, 22.9340477, -68.2838745, 68.2824249
25: -38.6347656, 31.1451912, -38.6372414, 31.1439552, -69.7787170, 69.7824249
26: -59.2232208, 37.7922516, -59.2214470, 37.7927284, -97.0159302, 97.0136948
27: -49.4987373, 27.4356384, -49.4956703, 27.4302959, -76.9290314, 76.9313049
28: -37.9599075, 28.9263287, -37.9574013, 28.9253578, -66.8852692, 66.8837280
29: -55.5999985, 34.4687195, -55.5970802, 34.4656448, -90.0656433, 90.0657959
30: -47.9058228, 27.3285789, -47.9027634, 27.3255386, -75.2313614, 75.2313309
31: -49.1463318, 24.1118011, -49.1433868, 24.1026268, -73.2489624, 73.2551880
32: -49.2434464, 27.5422554, -49.2425423, 27.5327263, -76.7761688, 76.7847977
33: -71.9991302, 44.1582756, -71.9976501, 44.1563950, -116.1555252, 116.1559219
34: -61.0185089, 30.1541843, -61.0157928, 30.1531982, -91.1717072, 91.1699677
35: -57.3175583, 34.8045311, -57.3073502, 34.8016586, -92.1192169, 92.1118774
36: -57.3812714, 34.0554962, -57.3817520, 34.0549889, -91.4362640, 91.4372482
37: -85.4045029, 33.2435570, -85.4009552, 33.2300568, -118.6345596, 118.6444931
38: -69.2060547, 41.0732727, -69.2046814, 41.0679779, -110.2740097, 110.2779388
39: -85.1687164, 40.8621216, -85.1664581, 40.8593712, -126.0280609, 126.0285645
40: -75.3106995, 30.0822411, -75.3065796, 30.0789909, -105.3896942, 105.3888245
41: -54.4104233, 26.0679359, -54.4086761, 26.0647087, -80.4751282, 80.4766083
42: -39.0048599, 29.5098457, -39.0034256, 29.5072498, -68.5121078, 68.5132675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9384794, upper bound: 39.0081453
time: 96.82 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8204510, upper bound: 38.8835263
time: 76.17 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 175.39 seconds
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 175.39
Output dim: 2, lower bound: -38.9384794, upper bound: 39.0081453
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 175.39
Output dim: 2, lower bound: -38.8204510, upper bound: 38.8835263

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -53.2908363, 42.9880066, -53.4391403, 43.0674324, -96.3582687, 96.4271393
1: -31.6059303, 36.0325851, -31.7082748, 36.0993614, -67.7052917, 67.7408600
2: -30.3753586, 35.5973015, -30.4800301, 35.6698570, -66.0452118, 66.0773315
3: -33.9193573, 41.5928192, -34.0228729, 41.6647720, -75.5841217, 75.6156921
4: -40.0356522, 38.9123611, -40.1256905, 38.9800720, -79.0157242, 79.0380402
5: -36.8690109, 41.3739471, -36.9697647, 41.4498711, -78.3188705, 78.3437119
6: -55.9693108, 22.4697437, -56.0080605, 22.5067997, -78.4761124, 78.4777985
7: -42.8674164, 40.1504478, -43.0070038, 40.2531738, -83.1205826, 83.1574554
8: -39.2895432, 45.4624977, -39.4390106, 45.5848961, -84.8744354, 84.9015045
9: -34.2102814, 37.5118027, -34.2572289, 37.5517349, -71.7620163, 71.7690277
10: -55.2226486, 52.4093857, -55.2895393, 52.4482155, -107.6708603, 107.6989212
11: -56.5202103, 39.7194672, -56.5958214, 39.7864265, -96.3066406, 96.3152924
12: -59.1667023, 44.0005112, -59.2368851, 44.1101570, -103.2768555, 103.2373962
13: -48.7971115, 49.6238785, -48.8302536, 49.6780014, -98.4750900, 98.4541321
14: -81.4359894, 43.3726082, -81.6030273, 43.4696274, -124.9056091, 124.9756317
15: -40.3837509, 36.3949966, -40.4519005, 36.4357605, -76.8195038, 76.8468933
16: -58.3212967, 40.8520660, -58.3939171, 40.9001503, -99.2214508, 99.2459869
17: -85.2348480, 62.5520210, -85.3231812, 62.6095352, -147.8443909, 147.8751984
18: -49.0153389, 29.1426620, -49.0875664, 29.1713314, -78.1866684, 78.2302246
19: -41.3974686, 19.4875813, -41.4605942, 19.5514507, -60.9489212, 60.9481735
20: -35.4334564, 21.8229523, -35.4723206, 21.8548374, -57.2882881, 57.2952728
21: -49.2191353, 25.4660378, -49.2822113, 25.5168037, -74.7359390, 74.7482452
22: -50.9810982, 30.0220680, -51.0842743, 30.1279697, -81.1090698, 81.1063385
23: -39.1725540, 26.6050320, -39.2468872, 26.6626167, -65.8351746, 65.8519211
24: -45.2244644, 22.8411102, -45.3223610, 22.9007950, -68.1252594, 68.1634674
25: -38.5205307, 30.9921722, -38.6151962, 31.0899220, -69.6104507, 69.6073685
26: -59.0800819, 37.5729828, -59.1961479, 37.7146606, -96.7947388, 96.7691269
27: -49.4248428, 27.3977604, -49.4750977, 27.4130459, -76.8378830, 76.8728561
28: -37.8757515, 28.8348389, -37.9450264, 28.8947258, -66.7704773, 66.7798615
29: -55.4521713, 34.3335419, -55.5654678, 34.4178085, -89.8699799, 89.8990097
30: -47.7844925, 27.2391701, -47.8793640, 27.2968082, -75.0812988, 75.1185303
31: -49.0453415, 24.0423851, -49.1239433, 24.0792274, -73.1245651, 73.1663208
32: -49.1908379, 27.4825916, -49.2274513, 27.5045719, -76.6954117, 76.7100372
33: -71.8902130, 43.9933624, -71.9791565, 44.0993156, -115.9895325, 115.9725189
34: -60.9175110, 30.0155048, -60.9983444, 30.1066818, -91.0241928, 91.0138474
35: -57.1994858, 34.6738434, -57.2892532, 34.7559814, -91.9554596, 91.9630966
36: -57.2970963, 33.9067764, -57.3663521, 34.0033264, -91.3004150, 91.2731247
37: -85.2404785, 33.0182533, -85.3765945, 33.1506271, -118.3911057, 118.3948364
38: -69.1388245, 41.0139389, -69.1856689, 41.0497360, -110.1885529, 110.1996078
39: -85.0803528, 40.7591782, -85.1430664, 40.8241119, -125.9044647, 125.9022369
40: -75.2387314, 30.0040512, -75.2885437, 30.0541325, -105.2928467, 105.2925949
41: -54.3328400, 25.9478760, -54.3968964, 26.0248566, -80.3576889, 80.3447723
42: -38.9623184, 29.4241085, -38.9942131, 29.4801044, -68.4424210, 68.4183197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8124373, upper bound: 38.9989777
time: 71.41 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8124373, upper bound: 39.0081459
time: 75.22 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 149.02 seconds
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 149.02
Output dim: 2, lower bound: -38.8124373, upper bound: 38.9989777
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 149.02
Output dim: 2, lower bound: -38.8124373, upper bound: 39.0081459

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -53.2908363, 42.9880066, -53.2972641, 42.9009399, -96.1917725, 96.2852707
1: -31.6059303, 36.0325851, -31.6104202, 35.9674377, -67.5733643, 67.6430054
2: -30.3753586, 35.5973015, -30.3769455, 35.4441071, -65.8194580, 65.9742432
3: -33.9193573, 41.5928192, -33.9301376, 41.4508896, -75.3702469, 75.5229568
4: -40.0356522, 38.9123611, -40.0306892, 38.7622185, -78.7978668, 78.9430542
5: -36.8690109, 41.3739471, -36.8734283, 41.1871643, -78.0561752, 78.2473755
6: -55.9693108, 22.4697437, -55.9433136, 22.4213600, -78.3906555, 78.4130554
7: -42.8674164, 40.1504478, -42.8457870, 39.9420242, -82.8094330, 82.9962311
8: -39.2895432, 45.4624977, -39.3365784, 45.3377838, -84.6273270, 84.7990723
9: -34.2102814, 37.5118027, -34.0943069, 37.4176941, -71.6279755, 71.6061096
10: -55.2226486, 52.4093857, -55.0257835, 52.2784195, -107.5010681, 107.4351654
11: -56.5202103, 39.7194672, -56.5076866, 39.7221146, -96.2423248, 96.2271500
12: -59.1667023, 44.0005112, -58.6891289, 43.8943939, -103.0610809, 102.6896362
13: -48.7971115, 49.6238785, -48.6009979, 49.4981270, -98.2952271, 98.2248688
14: -81.4359894, 43.3726082, -81.1351776, 43.3144684, -124.7504578, 124.5077820
15: -40.3837509, 36.3949966, -40.2667847, 36.3328400, -76.7165909, 76.6617813
16: -58.3212967, 40.8520660, -58.2243958, 40.7582817, -99.0795746, 99.0764618
17: -85.2348480, 62.5520210, -84.8638000, 62.4420242, -147.6768646, 147.4158173
18: -49.0153389, 29.1426620, -48.9166908, 29.1025658, -78.1179047, 78.0593567
19: -41.3974686, 19.4875813, -41.3452263, 19.5067215, -60.9041901, 60.8328094
20: -35.4334564, 21.8229523, -35.3387756, 21.7952003, -57.2286530, 57.1617203
21: -49.2191353, 25.4660378, -49.1465912, 25.4645500, -74.6836853, 74.6126175
22: -50.9810982, 30.0220680, -50.7666321, 29.9963875, -80.9774857, 80.7886887
23: -39.1725540, 26.6050320, -39.1265411, 26.6276436, -65.8002014, 65.7315750
24: -45.2244644, 22.8411102, -45.1949654, 22.7912292, -68.0156937, 68.0360718
25: -38.5205307, 30.9921722, -38.5085144, 31.0135288, -69.5340576, 69.5006866
26: -59.0800819, 37.5729828, -58.6854324, 37.5221214, -96.6022034, 96.2584152
27: -49.4248428, 27.3977604, -49.3232307, 27.3107147, -76.7355576, 76.7209930
28: -37.8757515, 28.8348389, -37.8351212, 28.8473148, -66.7230606, 66.6699600
29: -55.4521713, 34.3335419, -55.2774467, 34.2927551, -89.7449265, 89.6109924
30: -47.7844925, 27.2391701, -47.7858734, 27.2347298, -75.0192261, 75.0250397
31: -49.0453415, 24.0423851, -48.9575157, 23.9187088, -72.9640503, 72.9999008
32: -49.1908379, 27.4825916, -49.0708885, 27.4122772, -76.6031189, 76.5534821
33: -71.8902130, 43.9933624, -71.8464508, 43.9117126, -115.8019257, 115.8398132
34: -60.9175110, 30.0155048, -60.8938675, 30.0449028, -90.9624100, 90.9093704
35: -57.1994858, 34.6738434, -57.1873970, 34.6931458, -91.8926315, 91.8612366
36: -57.2970963, 33.9067764, -57.1560440, 33.8982620, -91.1953583, 91.0628204
37: -85.2404785, 33.0182533, -85.2276306, 33.0760345, -118.3165131, 118.2458801
38: -69.1388245, 41.0139389, -69.0205917, 40.9498672, -110.0886917, 110.0345306
39: -85.0803528, 40.7591782, -85.0320129, 40.6719513, -125.7523041, 125.7911911
40: -75.2387314, 30.0040512, -75.1640930, 29.8612480, -105.0999680, 105.1681366
41: -54.3328400, 25.9478760, -54.2964020, 25.9477730, -80.2806091, 80.2442780
42: -38.9623184, 29.4241085, -38.8457718, 29.3809319, -68.3432465, 68.2698746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 665

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7990206, upper bound: 38.9296085
time: 1454.61 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8064906, upper bound: 38.9964385
time: 172.19 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -53.2908363, 42.9880066, -53.4330750, 43.0569229, -96.3477478, 96.4210815
1: -31.6059303, 36.0325851, -31.7043533, 36.0904999, -67.6964188, 67.7369385
2: -30.3753586, 35.5973015, -30.4771252, 35.6655960, -66.0409546, 66.0744247
3: -33.9193573, 41.5928192, -34.0199051, 41.6599884, -75.5793381, 75.6127243
4: -40.0356522, 38.9123611, -40.1229706, 38.9745674, -79.0102234, 79.0353241
5: -36.8690109, 41.3739471, -36.9672012, 41.4446373, -78.3136444, 78.3411484
6: -55.9693108, 22.4697437, -56.0037155, 22.4942207, -78.4635315, 78.4734573
7: -42.8674164, 40.1504478, -43.0015640, 40.2465515, -83.1139679, 83.1520081
8: -39.2895432, 45.4624977, -39.4350204, 45.5797043, -84.8692474, 84.8975220
9: -34.2102814, 37.5118027, -34.2535515, 37.5476570, -71.7579346, 71.7653503
10: -55.2226486, 52.4093857, -55.2831612, 52.4440536, -107.6666946, 107.6925507
11: -56.5202103, 39.7194672, -56.5888138, 39.7724190, -96.2926331, 96.3082733
12: -59.1667023, 44.0005112, -59.2270279, 44.1069336, -103.2736359, 103.2275391
13: -48.7971115, 49.6238785, -48.8251724, 49.6722450, -98.4693527, 98.4490433
14: -81.4359894, 43.3726082, -81.5927429, 43.4673843, -124.9033737, 124.9653473
15: -40.3837509, 36.3949966, -40.4281540, 36.4305573, -76.8143005, 76.8231506
16: -58.3212967, 40.8520660, -58.3881798, 40.8718338, -99.1931305, 99.2402496
17: -85.2348480, 62.5520210, -85.3146362, 62.6054764, -147.8403320, 147.8666534
18: -49.0153389, 29.1426620, -49.0823555, 29.1678810, -78.1832123, 78.2250214
19: -41.3974686, 19.4875813, -41.4574890, 19.5413151, -60.9387741, 60.9450684
20: -35.4334564, 21.8229523, -35.4683609, 21.8531742, -57.2866211, 57.2913132
21: -49.2191353, 25.4660378, -49.2771225, 25.5132370, -74.7323761, 74.7431564
22: -50.9810982, 30.0220680, -51.0708847, 30.1227131, -81.1038132, 81.0929413
23: -39.1725540, 26.6050320, -39.2436523, 26.6591225, -65.8316727, 65.8486786
24: -45.2244644, 22.8411102, -45.3179512, 22.8978882, -68.1223526, 68.1590576
25: -38.5205307, 30.9921722, -38.6057968, 31.0861969, -69.6067276, 69.5979614
26: -59.0800819, 37.5729828, -59.1859016, 37.7095490, -96.7896271, 96.7588806
27: -49.4248428, 27.3977604, -49.4693375, 27.4087410, -76.8335800, 76.8670959
28: -37.8757515, 28.8348389, -37.9425964, 28.8923988, -66.7681351, 66.7774353
29: -55.4521713, 34.3335419, -55.5573463, 34.4143944, -89.8665619, 89.8908844
30: -47.7844925, 27.2391701, -47.8740005, 27.2948608, -75.0793533, 75.1131668
31: -49.0453415, 24.0423851, -49.1194153, 24.0752048, -73.1205444, 73.1617966
32: -49.1908379, 27.4825916, -49.2219696, 27.5020752, -76.6929092, 76.7045593
33: -71.8902130, 43.9933624, -71.9759216, 44.0940247, -115.9842377, 115.9692841
34: -60.9175110, 30.0155048, -60.9955597, 30.1031818, -91.0206909, 91.0110626
35: -57.1994858, 34.6738434, -57.2862968, 34.7529335, -91.9524231, 91.9601440
36: -57.2970963, 33.9067764, -57.3607788, 34.0001717, -91.2972565, 91.2675552
37: -85.2404785, 33.0182533, -85.3719177, 33.1472473, -118.3877258, 118.3901596
38: -69.1388245, 41.0139389, -69.1764908, 41.0467224, -110.1855316, 110.1904297
39: -85.0803528, 40.7591782, -85.1391754, 40.8197632, -125.9001160, 125.8983459
40: -75.2387314, 30.0040512, -75.2831421, 30.0500011, -105.2887115, 105.2871933
41: -54.3328400, 25.9478760, -54.3933907, 26.0149269, -80.3477554, 80.3412628
42: -38.9623184, 29.4241085, -38.9894409, 29.4773407, -68.4396591, 68.4135437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 665

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7990206, upper bound: 38.9393651
time: 79.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8064906, upper bound: 38.9964391
time: 83.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 166.01 seconds
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 166.01
Output dim: 2, lower bound: -38.7990206, upper bound: 38.9296085
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 166.01
Output dim: 2, lower bound: -38.8064906, upper bound: 38.9964385
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 166.01
Output dim: 2, lower bound: -38.7990206, upper bound: 38.9393651
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 166.01
Output dim: 2, lower bound: -38.8064906, upper bound: 38.9964391

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -53.2830963, 42.9738388, -53.2971725, 42.9007339, -96.1838226, 96.2710114
1: -31.6008263, 36.0192757, -31.6103573, 35.9672699, -67.5680847, 67.6296234
2: -30.3712559, 35.5915222, -30.3768826, 35.4440384, -65.8152924, 65.9683990
3: -33.9151573, 41.5852165, -33.9300728, 41.4507866, -75.3659439, 75.5152893
4: -40.0294037, 38.9048233, -40.0306053, 38.7621155, -78.7915192, 78.9354248
5: -36.8650360, 41.3631248, -36.8733864, 41.1870499, -78.0520782, 78.2365112
6: -55.9619141, 22.4626942, -55.9432220, 22.4212818, -78.3831940, 78.4059143
7: -42.8608208, 40.1190262, -42.8456841, 39.9416122, -82.8024216, 82.9647064
8: -39.2833672, 45.4548950, -39.3364944, 45.3376923, -84.6210632, 84.7913895
9: -34.1962776, 37.5068398, -34.0941200, 37.4176331, -71.6139069, 71.6009598
10: -55.1995926, 52.4024315, -55.0254898, 52.2783394, -107.4779358, 107.4279175
11: -56.5114479, 39.7118378, -56.5075836, 39.7220306, -96.2334747, 96.2194214
12: -59.1531219, 43.9940186, -58.6889572, 43.8943062, -103.0474243, 102.6829758
13: -48.7876816, 49.6157951, -48.6008911, 49.4980164, -98.2856979, 98.2166748
14: -81.4207077, 43.3679504, -81.1350021, 43.3144188, -124.7351227, 124.5029449
15: -40.3357239, 36.3870277, -40.2660942, 36.3327408, -76.6684647, 76.6531143
16: -58.3084793, 40.8129349, -58.2242355, 40.7577095, -99.0661850, 99.0371704
17: -85.2208557, 62.5439758, -84.8636398, 62.4418793, -147.6627350, 147.4076080
18: -49.0008430, 29.1342373, -48.9165115, 29.1024647, -78.1033096, 78.0507431
19: -41.3902740, 19.4624748, -41.3451309, 19.5064182, -60.8966866, 60.8076019
20: -35.4264450, 21.8196545, -35.3386841, 21.7951584, -57.2216034, 57.1583328
21: -49.2111359, 25.4597378, -49.1464882, 25.4644756, -74.6756134, 74.6062241
22: -50.9652519, 30.0147343, -50.7664413, 29.9962826, -80.9615326, 80.7811737
23: -39.1667557, 26.6011009, -39.1264725, 26.6275978, -65.7943573, 65.7275696
24: -45.2191658, 22.8362560, -45.1948853, 22.7911644, -68.0103226, 68.0311432
25: -38.5086288, 30.9870605, -38.5083580, 31.0134716, -69.5221024, 69.4954224
26: -59.0659485, 37.5640945, -58.6852570, 37.5220146, -96.5879669, 96.2493515
27: -49.4180679, 27.3878479, -49.3231392, 27.3105774, -76.7286377, 76.7109833
28: -37.8703804, 28.8309689, -37.8350525, 28.8472595, -66.7176361, 66.6660233
29: -55.4415855, 34.3284607, -55.2773209, 34.2926865, -89.7342682, 89.6057816
30: -47.7783470, 27.2346458, -47.7857895, 27.2346764, -75.0130234, 75.0204315
31: -49.0367813, 24.0153923, -48.9574242, 23.9183502, -72.9551315, 72.9728165
32: -49.1837616, 27.4774361, -49.0707970, 27.4122162, -76.5959625, 76.5482330
33: -71.8822327, 43.9840660, -71.8463516, 43.9116058, -115.7938385, 115.8304138
34: -60.9130020, 30.0079899, -60.8937912, 30.0448189, -90.9578171, 90.9017792
35: -57.1936111, 34.6678467, -57.1873360, 34.6930466, -91.8866501, 91.8551788
36: -57.2907295, 33.9014091, -57.1559601, 33.8981934, -91.1889114, 91.0573730
37: -85.2312241, 33.0083160, -85.2275085, 33.0758743, -118.3070984, 118.2358246
38: -69.1297684, 41.0066757, -69.0204773, 40.9497681, -110.0795364, 110.0271454
39: -85.0703125, 40.7506485, -85.0318756, 40.6718292, -125.7421417, 125.7825165
40: -75.2302246, 29.9972591, -75.1639709, 29.8611679, -105.0913849, 105.1612244
41: -54.3264847, 25.9293518, -54.2963257, 25.9475117, -80.2739868, 80.2256775
42: -38.9555054, 29.4172592, -38.8456917, 29.3808556, -68.3363571, 68.2629547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 665

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6421613, upper bound: 38.9904299
time: 87.46 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6421613, upper bound: 38.9964390
time: 71.50 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -53.2830963, 42.9738388, -53.4329834, 43.0567551, -96.3398438, 96.4068222
1: -31.6008263, 36.0192757, -31.7042999, 36.0903549, -67.6911774, 67.7235718
2: -30.3712559, 35.5915222, -30.4770699, 35.6655273, -66.0367813, 66.0685883
3: -33.9151573, 41.5852165, -34.0198746, 41.6598892, -75.5750427, 75.6050873
4: -40.0294037, 38.9048233, -40.1229057, 38.9744720, -79.0038757, 79.0277252
5: -36.8650360, 41.3631248, -36.9671326, 41.4445000, -78.3095398, 78.3302612
6: -55.9619141, 22.4626942, -56.0036240, 22.4941349, -78.4560471, 78.4663162
7: -42.8608208, 40.1190262, -43.0014839, 40.2461472, -83.1069641, 83.1205063
8: -39.2833672, 45.4548950, -39.4349365, 45.5796242, -84.8629913, 84.8898315
9: -34.1962776, 37.5068398, -34.2533684, 37.5475922, -71.7438660, 71.7602081
10: -55.1995926, 52.4024315, -55.2828941, 52.4439774, -107.6435547, 107.6853256
11: -56.5114479, 39.7118378, -56.5887146, 39.7723160, -96.2837677, 96.3005524
12: -59.1531219, 43.9940186, -59.2268677, 44.1068420, -103.2599487, 103.2208862
13: -48.7876816, 49.6157951, -48.8250504, 49.6721611, -98.4598389, 98.4408417
14: -81.4207077, 43.3679504, -81.5925598, 43.4673195, -124.8880310, 124.9605103
15: -40.3357239, 36.3870277, -40.4275322, 36.4304733, -76.7661896, 76.8145523
16: -58.3084793, 40.8129349, -58.3880424, 40.8713455, -99.1798248, 99.2009735
17: -85.2208557, 62.5439758, -85.3144684, 62.6053886, -147.8262329, 147.8584442
18: -49.0008430, 29.1342373, -49.0821991, 29.1677628, -78.1686096, 78.2164307
19: -41.3902740, 19.4624748, -41.4574051, 19.5410194, -60.9312935, 60.9198799
20: -35.4264450, 21.8196545, -35.4682693, 21.8531399, -57.2795830, 57.2879181
21: -49.2111359, 25.4597378, -49.2770309, 25.5131626, -74.7242889, 74.7367706
22: -50.9652519, 30.0147343, -51.0707016, 30.1226254, -81.0878754, 81.0854340
23: -39.1667557, 26.6011009, -39.2435799, 26.6590767, -65.8258286, 65.8446808
24: -45.2191658, 22.8362560, -45.3178902, 22.8978291, -68.1169891, 68.1541443
25: -38.5086288, 30.9870605, -38.6056290, 31.0861359, -69.5947571, 69.5926895
26: -59.0659485, 37.5640945, -59.1857300, 37.7094345, -96.7753830, 96.7498169
27: -49.4180679, 27.3878479, -49.4692497, 27.4086189, -76.8266830, 76.8570938
28: -37.8703804, 28.8309689, -37.9425163, 28.8923454, -66.7627258, 66.7734833
29: -55.4415855, 34.3284607, -55.5572205, 34.4143257, -89.8559113, 89.8856812
30: -47.7783470, 27.2346458, -47.8739319, 27.2947998, -75.0731430, 75.1085815
31: -49.0367813, 24.0153923, -49.1193161, 24.0748520, -73.1116333, 73.1347046
32: -49.1837616, 27.4774361, -49.2218895, 27.5020084, -76.6857681, 76.6993256
33: -71.8822327, 43.9840660, -71.9758301, 44.0939064, -115.9761353, 115.9598999
34: -60.9130020, 30.0079899, -60.9954987, 30.1030846, -91.0160828, 91.0034866
35: -57.1936111, 34.6678467, -57.2862129, 34.7528381, -91.9464493, 91.9540558
36: -57.2907295, 33.9014091, -57.3607025, 34.0001068, -91.2908173, 91.2621078
37: -85.2312241, 33.0083160, -85.3717957, 33.1471062, -118.3783264, 118.3801117
38: -69.1297684, 41.0066757, -69.1763687, 41.0466423, -110.1764069, 110.1830444
39: -85.0703125, 40.7506485, -85.1390610, 40.8196373, -125.8899536, 125.8897095
40: -75.2302246, 29.9972591, -75.2830353, 30.0499229, -105.2801514, 105.2802887
41: -54.3264847, 25.9293518, -54.3933105, 26.0147018, -80.3411713, 80.3226624
42: -38.9555054, 29.4172592, -38.9893646, 29.4772453, -68.4327469, 68.4066238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 665

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7716908, upper bound: 39.0021093
time: 77.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6421613, upper bound: 39.0021087
time: 142.51 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 222.81 seconds
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 222.81
Output dim: 2, lower bound: -38.6421613, upper bound: 38.9904299
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 222.81
Output dim: 2, lower bound: -38.6421613, upper bound: 38.9964390
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 222.81
Output dim: 2, lower bound: -38.7716908, upper bound: 39.0021093
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 222.81
Output dim: 2, lower bound: -38.6421613, upper bound: 39.0021087

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -53.2830963, 42.9738388, -53.1423607, 42.7061386, -95.9892349, 96.1161957
1: -31.6008263, 36.0192757, -31.5036354, 35.8115158, -67.4123306, 67.5229111
2: -30.3712559, 35.5915222, -30.2635632, 35.2125931, -65.5838470, 65.8550873
3: -33.9151573, 41.5852165, -33.8345566, 41.2085571, -75.1237183, 75.4197693
4: -40.0294037, 38.9048233, -39.9036942, 38.4507523, -78.4801483, 78.8085175
5: -36.8650360, 41.3631248, -36.8085022, 40.9567184, -77.8217392, 78.1716309
6: -55.9619141, 22.4626942, -55.7770805, 22.2740479, -78.2359619, 78.2397766
7: -42.8608208, 40.1190262, -42.7198715, 39.6769562, -82.5377808, 82.8388977
8: -39.2833672, 45.4548950, -39.2119217, 45.0250130, -84.3083801, 84.6668167
9: -34.1962776, 37.5068398, -34.0050735, 37.2796936, -71.4759674, 71.5119019
10: -55.1995926, 52.4024315, -54.8070908, 52.0954361, -107.2950287, 107.2095184
11: -56.5114479, 39.7118378, -56.3219376, 39.6317444, -96.1431885, 96.0337753
12: -59.1531219, 43.9940186, -58.0177460, 43.6397476, -102.7928619, 102.0117569
13: -48.7876816, 49.6157951, -48.3497238, 49.3063622, -98.0940399, 97.9655151
14: -81.4207077, 43.3679504, -80.5199280, 43.1249580, -124.5456619, 123.8878708
15: -40.3357239, 36.3870277, -40.1028214, 36.3240280, -76.6597519, 76.4898376
16: -58.3084793, 40.8129349, -58.0052376, 40.5641556, -98.8726349, 98.8181763
17: -85.2208557, 62.5439758, -84.1158295, 62.2047043, -147.4255371, 146.6598053
18: -49.0008430, 29.1342373, -48.6709595, 29.0001163, -78.0009613, 77.8051910
19: -41.3902740, 19.4624748, -41.2463455, 19.4657822, -60.8560562, 60.7088165
20: -35.4264450, 21.8196545, -35.1283722, 21.7043419, -57.1307869, 56.9480247
21: -49.2111359, 25.4597378, -48.9750671, 25.3994904, -74.6106262, 74.4348068
22: -50.9652519, 30.0147343, -50.3182335, 29.8429241, -80.8081665, 80.3329697
23: -39.1667557, 26.6011009, -38.9640808, 26.5898933, -65.7566528, 65.5651855
24: -45.2191658, 22.8362560, -45.0295868, 22.6668854, -67.8860474, 67.8658447
25: -38.5086288, 30.9870605, -38.3267632, 30.9115944, -69.4202271, 69.3138199
26: -59.0659485, 37.5640945, -58.0407104, 37.3459015, -96.4118500, 95.6048050
27: -49.4180679, 27.3878479, -49.1575203, 27.2239418, -76.6420135, 76.5453644
28: -37.8703804, 28.8309689, -37.7123871, 28.7902832, -66.6606598, 66.5433502
29: -55.4415855, 34.3284607, -54.7751160, 34.1382332, -89.5798187, 89.1035767
30: -47.7783470, 27.2346458, -47.6670914, 27.1450195, -74.9233627, 74.9017334
31: -49.0367813, 24.0153923, -48.8207588, 23.8038254, -72.8406067, 72.8361511
32: -49.1837616, 27.4774361, -48.8760109, 27.2851582, -76.4689178, 76.3534393
33: -71.8822327, 43.9840660, -71.6452484, 43.5343513, -115.4165802, 115.6293030
34: -60.9130020, 30.0079899, -60.7545471, 29.8899231, -90.8029251, 90.7625351
35: -57.1936111, 34.6678467, -57.0265388, 34.5146866, -91.7082977, 91.6943817
36: -57.2907295, 33.9014091, -56.8773994, 33.7698631, -91.0605927, 90.7788086
37: -85.2312241, 33.0083160, -85.0030212, 32.9253464, -118.1565628, 118.0113373
38: -69.1297684, 41.0066757, -68.7572403, 40.7906876, -109.9204559, 109.7639160
39: -85.0703125, 40.7506485, -84.8526306, 40.2947617, -125.3650665, 125.6032791
40: -75.2302246, 29.9972591, -74.9981766, 29.5409470, -104.7711716, 104.9954376
41: -54.3264847, 25.9293518, -54.1826401, 25.8435745, -80.1700592, 80.1119919
42: -38.9555054, 29.4172592, -38.6275024, 29.2384338, -68.1939392, 68.0447617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=208, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=405, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 648

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7449248, upper bound: 38.8390338
time: 75.15 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6345970, upper bound: 38.8390338
time: 74.04 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -53.2830963, 42.9738388, -53.2892761, 42.8848190, -96.1679153, 96.2631149
1: -31.6008263, 36.0192757, -31.6051769, 35.9531631, -67.5539780, 67.6244507
2: -30.3712559, 35.5915222, -30.3726845, 35.4382935, -65.8095398, 65.9642029
3: -33.9151573, 41.5852165, -33.9257164, 41.4430771, -75.3582306, 75.5109253
4: -40.0294037, 38.9048233, -40.0244675, 38.7546082, -78.7840118, 78.9292831
5: -36.8650360, 41.3631248, -36.8692932, 41.1764336, -78.0414734, 78.2324066
6: -55.9619141, 22.4626942, -55.9358635, 22.4142265, -78.3761444, 78.3985596
7: -42.8608208, 40.1190262, -42.8388748, 39.9094353, -82.7702560, 82.9578934
8: -39.2833672, 45.4548950, -39.3302994, 45.3301430, -84.6135101, 84.7851868
9: -34.1962776, 37.5068398, -34.0800133, 37.4125938, -71.6088638, 71.5868454
10: -55.1995926, 52.4024315, -55.0026245, 52.2712555, -107.4708405, 107.4050522
11: -56.5114479, 39.7118378, -56.4989433, 39.7144547, -96.2258987, 96.2107773
12: -59.1531219, 43.9940186, -58.6755371, 43.8877449, -103.0408630, 102.6695404
13: -48.7876816, 49.6157951, -48.5915489, 49.4898262, -98.2775116, 98.2073364
14: -81.4207077, 43.3679504, -81.1196747, 43.3097839, -124.7304764, 124.4876251
15: -40.3357239, 36.3870277, -40.2117538, 36.3246994, -76.6604233, 76.5987778
16: -58.3084793, 40.8129349, -58.2116585, 40.7152748, -99.0237579, 99.0245972
17: -85.2208557, 62.5439758, -84.8497314, 62.4338684, -147.6547089, 147.3937073
18: -49.0008430, 29.1342373, -48.9023552, 29.0940380, -78.0948792, 78.0365906
19: -41.3902740, 19.4624748, -41.3380394, 19.4813004, -60.8715630, 60.8005142
20: -35.4264450, 21.8196545, -35.3317070, 21.7918377, -57.2182846, 57.1513596
21: -49.2111359, 25.4597378, -49.1384697, 25.4586182, -74.6697540, 74.5982056
22: -50.9652519, 30.0147343, -50.7505112, 29.9888229, -80.9540710, 80.7652435
23: -39.1667557, 26.6011009, -39.1206779, 26.6237183, -65.7904739, 65.7217789
24: -45.2191658, 22.8362560, -45.1894073, 22.7863426, -68.0055008, 68.0256577
25: -38.5086288, 30.9870605, -38.4962463, 31.0083218, -69.5169525, 69.4833069
26: -59.0659485, 37.5640945, -58.6711807, 37.5128632, -96.5788116, 96.2352753
27: -49.4180679, 27.3878479, -49.3162193, 27.3003426, -76.7184067, 76.7040558
28: -37.8703804, 28.8309689, -37.8297272, 28.8433704, -66.7137527, 66.6606903
29: -55.4415855, 34.3284607, -55.2665596, 34.2874870, -89.7290726, 89.5950165
30: -47.7783470, 27.2346458, -47.7795029, 27.2302437, -75.0085907, 75.0141449
31: -49.0367813, 24.0153923, -48.9488602, 23.8915405, -72.9283142, 72.9642487
32: -49.1837616, 27.4774361, -49.0637054, 27.4070740, -76.5908356, 76.5411377
33: -71.8822327, 43.9840660, -71.8384857, 43.9024773, -115.7847061, 115.8225555
34: -60.9130020, 30.0079899, -60.8891907, 30.0374031, -90.9504089, 90.8971786
35: -57.1936111, 34.6678467, -57.1813965, 34.6871605, -91.8807678, 91.8492432
36: -57.2907295, 33.9014091, -57.1495552, 33.8926926, -91.1834106, 91.0509644
37: -85.2312241, 33.0083160, -85.2183685, 33.0656891, -118.2969131, 118.2266846
38: -69.1297684, 41.0066757, -69.0114975, 40.9426575, -110.0724258, 110.0181656
39: -85.0703125, 40.7506485, -85.0220337, 40.6636276, -125.7339401, 125.7726822
40: -75.2302246, 29.9972591, -75.1555786, 29.8544273, -105.0846481, 105.1528397
41: -54.3264847, 25.9293518, -54.2900391, 25.9282494, -80.2547302, 80.2193832
42: -38.9555054, 29.4172592, -38.8389282, 29.3739586, -68.3294678, 68.2561874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=208, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 648

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7449248, upper bound: 38.8487330
time: 82.12 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7524224, upper bound: 38.9949003
time: 82.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -53.2830963, 42.9738388, -53.2720718, 42.8586197, -96.1417160, 96.2459106
1: -31.6008263, 36.0192757, -31.5929089, 35.9292793, -67.5300903, 67.6121826
2: -30.3712559, 35.5915222, -30.3626461, 35.4318810, -65.8031311, 65.9541702
3: -33.9151573, 41.5852165, -33.9228783, 41.4116592, -75.3268127, 75.5080948
4: -40.0294037, 38.9048233, -39.9944344, 38.6593094, -78.6887131, 78.8992538
5: -36.8650360, 41.3631248, -36.9005737, 41.2113266, -78.0763550, 78.2636948
6: -55.9619141, 22.4626942, -55.8325729, 22.3456612, -78.3075714, 78.2952652
7: -42.8608208, 40.1190262, -42.8716240, 39.9693604, -82.8301849, 82.9906464
8: -39.2833672, 45.4548950, -39.3090973, 45.2639656, -84.5473328, 84.7639923
9: -34.1962776, 37.5068398, -34.1616554, 37.4036484, -71.5999222, 71.6684952
10: -55.1995926, 52.4024315, -55.0577278, 52.2596550, -107.4592361, 107.4601593
11: -56.5114479, 39.7118378, -56.3973808, 39.6811600, -96.1926117, 96.1092072
12: -59.1531219, 43.9940186, -58.5521927, 43.8518753, -103.0049973, 102.5462112
13: -48.7876816, 49.6157951, -48.5705147, 49.4760590, -98.2637329, 98.1863098
14: -81.4207077, 43.3679504, -80.9716797, 43.2772827, -124.6979752, 124.3396301
15: -40.3357239, 36.3870277, -40.2466240, 36.4023628, -76.7380829, 76.6336517
16: -58.3084793, 40.8129349, -58.1611176, 40.6515427, -98.9600220, 98.9740524
17: -85.2208557, 62.5439758, -84.5620499, 62.3665695, -147.5874023, 147.1060181
18: -49.0008430, 29.1342373, -48.8292427, 29.0632973, -78.0641327, 77.9634705
19: -41.3902740, 19.4624748, -41.3508987, 19.4980965, -60.8883667, 60.8133736
20: -35.4264450, 21.8196545, -35.2545471, 21.7614193, -57.1878624, 57.0741959
21: -49.2111359, 25.4597378, -49.1003799, 25.4470825, -74.6582184, 74.5601196
22: -50.9652519, 30.0147343, -50.6042938, 29.9661427, -80.9313889, 80.6190262
23: -39.1667557, 26.6011009, -39.0791550, 26.6199265, -65.7866745, 65.6802521
24: -45.2191658, 22.8362560, -45.1478386, 22.7712288, -67.9903946, 67.9840927
25: -38.5086288, 30.9870605, -38.4207458, 30.9800797, -69.4887085, 69.4078064
26: -59.0659485, 37.5640945, -58.5366592, 37.5316734, -96.5976257, 96.1007538
27: -49.4180679, 27.3878479, -49.2973366, 27.3190575, -76.7371216, 76.6851807
28: -37.8703804, 28.8309689, -37.8171692, 28.8334751, -66.7038574, 66.6481400
29: -55.4415855, 34.3284607, -55.0506401, 34.2596436, -89.7012329, 89.3791046
30: -47.7783470, 27.2346458, -47.7514114, 27.2043381, -74.9826813, 74.9860535
31: -49.0367813, 24.0153923, -48.9796448, 23.9561882, -72.9929657, 72.9950333
32: -49.1837616, 27.4774361, -49.0223312, 27.3722305, -76.5559845, 76.4997635
33: -71.8822327, 43.9840660, -71.7726135, 43.7137413, -115.5959625, 115.7566757
34: -60.9130020, 30.0079899, -60.8541908, 29.9439831, -90.8569870, 90.8621826
35: -57.1936111, 34.6678467, -57.1231689, 34.5683174, -91.7619324, 91.7910156
36: -57.2907295, 33.9014091, -57.0775490, 33.8687592, -91.1594849, 90.9789581
37: -85.2312241, 33.0083160, -85.1446075, 32.9917221, -118.2229309, 118.1529236
38: -69.1297684, 41.0066757, -68.9064789, 40.8849754, -110.0147324, 109.9131546
39: -85.0703125, 40.7506485, -84.9561691, 40.4325829, -125.5028992, 125.7068176
40: -75.2302246, 29.9972591, -75.1141968, 29.7272472, -104.9574738, 105.1114578
41: -54.3264847, 25.9293518, -54.2770920, 25.9087029, -80.2351837, 80.2064362
42: -38.9555054, 29.4172592, -38.7671509, 29.3333855, -68.2888947, 68.1844025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=208, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=405, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 648

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7672880, upper bound: 38.8525113
time: 78.28 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6345970, upper bound: 38.8525108
time: 197.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -53.2830963, 42.9738388, -53.4253502, 43.0427818, -96.3258667, 96.3991852
1: -31.6008263, 36.0192757, -31.6992569, 36.0771713, -67.6779938, 67.7185364
2: -30.3712559, 35.5915222, -30.4730225, 35.6598053, -66.0310516, 66.0645447
3: -33.9151573, 41.5852165, -34.0156975, 41.6523590, -75.5675049, 75.6009140
4: -40.0294037, 38.9048233, -40.1167297, 38.9670601, -78.9964600, 79.0215454
5: -36.8650360, 41.3631248, -36.9632187, 41.4338036, -78.2988281, 78.3263397
6: -55.9619141, 22.4626942, -55.9963150, 22.4871826, -78.4490967, 78.4590073
7: -42.8608208, 40.1190262, -42.9949646, 40.2151260, -83.0759430, 83.1139908
8: -39.2833672, 45.4548950, -39.4288254, 45.5721207, -84.8554840, 84.8837204
9: -34.1962776, 37.5068398, -34.2395325, 37.5426941, -71.7389679, 71.7463684
10: -55.1995926, 52.4024315, -55.2600822, 52.4370842, -107.6366730, 107.6625137
11: -56.5114479, 39.7118378, -56.5800629, 39.7647896, -96.2762375, 96.2918930
12: -59.1531219, 43.9940186, -59.2134438, 44.1004715, -103.2535934, 103.2074585
13: -48.7876816, 49.6157951, -48.8157234, 49.6641655, -98.4518433, 98.4315033
14: -81.4207077, 43.3679504, -81.5774841, 43.4627075, -124.8834076, 124.9454269
15: -40.3357239, 36.3870277, -40.3800507, 36.4225807, -76.7583008, 76.7670746
16: -58.3084793, 40.8129349, -58.3754082, 40.8324432, -99.1409225, 99.1883392
17: -85.2208557, 62.5439758, -85.3006439, 62.5973778, -147.8182373, 147.8446198
18: -49.0008430, 29.1342373, -49.0678444, 29.1594505, -78.1602936, 78.2020798
19: -41.3902740, 19.4624748, -41.4502869, 19.5161991, -60.9064674, 60.9127579
20: -35.4264450, 21.8196545, -35.4613495, 21.8498917, -57.2763367, 57.2809982
21: -49.2111359, 25.4597378, -49.2691307, 25.5069427, -74.7180786, 74.7288666
22: -50.9652519, 30.0147343, -51.0550117, 30.1154022, -81.0806427, 81.0697479
23: -39.1667557, 26.6011009, -39.2378845, 26.6551819, -65.8219299, 65.8389893
24: -45.2191658, 22.8362560, -45.3126602, 22.8930607, -68.1122284, 68.1489105
25: -38.5086288, 30.9870605, -38.5938873, 31.0810699, -69.5896988, 69.5809479
26: -59.0659485, 37.5640945, -59.1717453, 37.7006531, -96.7666016, 96.7358398
27: -49.4180679, 27.3878479, -49.4625282, 27.3988457, -76.8169098, 76.8503723
28: -37.8703804, 28.8309689, -37.9372215, 28.8885384, -66.7589188, 66.7681885
29: -55.4415855, 34.3284607, -55.5467873, 34.4093170, -89.8508911, 89.8752441
30: -47.7783470, 27.2346458, -47.8678665, 27.2903423, -75.0686874, 75.1025085
31: -49.0367813, 24.0153923, -49.1108437, 24.0481567, -73.0849380, 73.1262360
32: -49.1837616, 27.4774361, -49.2148972, 27.4969139, -76.6806717, 76.6923294
33: -71.8822327, 43.9840660, -71.9679565, 44.0847015, -115.9669342, 115.9520264
34: -60.9130020, 30.0079899, -60.9910660, 30.0956383, -91.0086365, 90.9990540
35: -57.1936111, 34.6678467, -57.2803650, 34.7469482, -91.9405518, 91.9482040
36: -57.2907295, 33.9014091, -57.3543816, 33.9947853, -91.2855072, 91.2557831
37: -85.2312241, 33.0083160, -85.3626175, 33.1372604, -118.3684845, 118.3709335
38: -69.1297684, 41.0066757, -69.1674194, 41.0394897, -110.1692581, 110.1740952
39: -85.0703125, 40.7506485, -85.1291504, 40.8112183, -125.8815308, 125.8797913
40: -75.2302246, 29.9972591, -75.2746429, 30.0432243, -105.2734528, 105.2718964
41: -54.3264847, 25.9293518, -54.3870392, 25.9964104, -80.3228836, 80.3163910
42: -38.9555054, 29.4172592, -38.9826202, 29.4705009, -68.4260025, 68.3998795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 648

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6345970, upper bound: 38.7045539
time: 106.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7701449, upper bound: 39.0005736
time: 83.73 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 192.26 seconds
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 192.26
Output dim: 2, lower bound: -38.7449248, upper bound: 38.8390338
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 192.26
Output dim: 2, lower bound: -38.6345970, upper bound: 38.8390338
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 192.26
Output dim: 2, lower bound: -38.7449248, upper bound: 38.8487330
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 192.26
Output dim: 2, lower bound: -38.7524224, upper bound: 38.9949003
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 192.26
Output dim: 2, lower bound: -38.7672880, upper bound: 38.8525113
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 192.26
Output dim: 2, lower bound: -38.6345970, upper bound: 38.8525108
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 192.26
Output dim: 2, lower bound: -38.6345970, upper bound: 38.7045539
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 192.26
Output dim: 2, lower bound: -38.7701449, upper bound: 39.0005736

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -53.2780838, 42.9685135, -53.2888336, 42.8843575, -96.1624451, 96.2573471
1: -31.5960712, 36.0158310, -31.6047268, 35.9528465, -67.5489197, 67.6205521
2: -30.3693523, 35.5874329, -30.3725109, 35.4379272, -65.8072739, 65.9599457
3: -33.9134293, 41.5795937, -33.9255753, 41.4425888, -75.3560181, 75.5051727
4: -40.0268555, 38.8998184, -40.0242157, 38.7541618, -78.7810135, 78.9240265
5: -36.8635330, 41.3571777, -36.8691635, 41.1759262, -78.0394592, 78.2263336
6: -55.9565392, 22.4434738, -55.9354134, 22.4124546, -78.3689957, 78.3788757
7: -42.8535004, 40.1130905, -42.8382072, 39.9089127, -82.7624130, 82.9512939
8: -39.2788544, 45.4493217, -39.3299026, 45.3296318, -84.6084900, 84.7792206
9: -34.1905670, 37.5037994, -34.0795212, 37.4123154, -71.6028748, 71.5833206
10: -55.1930618, 52.3995781, -55.0020828, 52.2710152, -107.4640808, 107.4016571
11: -56.5027313, 39.6936340, -56.4982185, 39.7129364, -96.2156677, 96.1918488
12: -59.1433868, 43.9921150, -58.6747055, 43.8875656, -103.0309448, 102.6668243
13: -48.7805481, 49.6053772, -48.5907669, 49.4888649, -98.2694092, 98.1961365
14: -81.4120483, 43.3646088, -81.1189423, 43.3094711, -124.7215195, 124.4835510
15: -40.3138199, 36.3796997, -40.2099075, 36.3240700, -76.6378937, 76.5895996
16: -58.3007126, 40.7951088, -58.2109642, 40.7129288, -99.0136414, 99.0060730
17: -85.2140808, 62.5395203, -84.8491058, 62.4334793, -147.6475525, 147.3886261
18: -48.9936676, 29.1244507, -48.9017143, 29.0932274, -78.0868835, 78.0261612
19: -41.3864365, 19.4548321, -41.3377151, 19.4806480, -60.8670845, 60.7925415
20: -35.4219284, 21.8182106, -35.3313217, 21.7917175, -57.2136421, 57.1495323
21: -49.2053719, 25.4582863, -49.1379700, 25.4584713, -74.6638412, 74.5962524
22: -50.9541969, 30.0072842, -50.7492142, 29.9881477, -80.9423370, 80.7564926
23: -39.1634941, 26.5992012, -39.1203995, 26.6235542, -65.7870483, 65.7196045
24: -45.2154579, 22.8342056, -45.1890869, 22.7861614, -68.0016174, 68.0232925
25: -38.5025406, 30.9809818, -38.4955864, 31.0077801, -69.5103226, 69.4765625
26: -59.0563698, 37.5599747, -58.6703377, 37.5124741, -96.5688477, 96.2303162
27: -49.4107590, 27.3830566, -49.3155403, 27.2999115, -76.7106705, 76.6985931
28: -37.8677826, 28.8281403, -37.8294945, 28.8431244, -66.7109070, 66.6576309
29: -55.4337120, 34.3235435, -55.2658577, 34.2870445, -89.7207565, 89.5893936
30: -47.7725449, 27.2328606, -47.7789993, 27.2300758, -75.0026093, 75.0118561
31: -49.0315819, 24.0076714, -48.9483948, 23.8907909, -72.9223709, 72.9560623
32: -49.1784325, 27.4757233, -49.0632553, 27.4069157, -76.5853424, 76.5389786
33: -71.8791351, 43.9780807, -71.8382111, 43.9019547, -115.7810898, 115.8162918
34: -60.9104233, 30.0039139, -60.8889618, 30.0370445, -90.9474640, 90.8928757
35: -57.1844292, 34.6630745, -57.1806335, 34.6867599, -91.8711853, 91.8437042
36: -57.2830467, 33.8969116, -57.1487083, 33.8922729, -91.1753235, 91.0456161
37: -85.2246399, 33.0039864, -85.2177963, 33.0653152, -118.2899551, 118.2217865
38: -69.1253052, 41.0039635, -69.0111160, 40.9424362, -110.0677414, 110.0150757
39: -85.0648727, 40.7467422, -85.0215454, 40.6633568, -125.7282257, 125.7682800
40: -75.2217407, 29.9927025, -75.1548157, 29.8538837, -105.0756226, 105.1475220
41: -54.3214188, 25.9177704, -54.2895927, 25.9271107, -80.2485275, 80.2073593
42: -38.9512062, 29.4150162, -38.8385582, 29.3737526, -68.3249588, 68.2535706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=208, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 649

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6773763, upper bound: 38.9934962
time: 72.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7617156, upper bound: 38.9935206
time: 65.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -53.2780838, 42.9685135, -53.4249115, 43.0423317, -96.3204193, 96.3934250
1: -31.5960712, 36.0158310, -31.6988354, 36.0768738, -67.6729431, 67.7146606
2: -30.3693523, 35.5874329, -30.4728546, 35.6594505, -66.0288010, 66.0602875
3: -33.9134293, 41.5795937, -34.0155563, 41.6518784, -75.5652924, 75.5951385
4: -40.0268555, 38.8998184, -40.1165199, 38.9666214, -78.9934769, 79.0163269
5: -36.8635330, 41.3571777, -36.9630890, 41.4332848, -78.2968140, 78.3202667
6: -55.9565392, 22.4434738, -55.9958496, 22.4855156, -78.4420547, 78.4393158
7: -42.8535004, 40.1130905, -42.9943314, 40.2145996, -83.0681000, 83.1074219
8: -39.2788544, 45.4493217, -39.4284477, 45.5716248, -84.8504791, 84.8777695
9: -34.1905670, 37.5037994, -34.2390366, 37.5424194, -71.7329865, 71.7428284
10: -55.1930618, 52.3995781, -55.2595291, 52.4368439, -107.6299057, 107.6591034
11: -56.5027313, 39.6936340, -56.5792961, 39.7632141, -96.2659454, 96.2729340
12: -59.1433868, 43.9921150, -59.2125854, 44.1002960, -103.2436676, 103.2046967
13: -48.7805481, 49.6053772, -48.8149261, 49.6632500, -98.4437866, 98.4202957
14: -81.4120483, 43.3646088, -81.5767365, 43.4624176, -124.8744659, 124.9413452
15: -40.3138199, 36.3796997, -40.3782730, 36.4219398, -76.7357635, 76.7579727
16: -58.3007126, 40.7951088, -58.3747025, 40.8309402, -99.1316452, 99.1698074
17: -85.2140808, 62.5395203, -85.3000488, 62.5970154, -147.8110962, 147.8395691
18: -48.9936676, 29.1244507, -49.0672073, 29.1586266, -78.1522980, 78.1916504
19: -41.3864365, 19.4548321, -41.4499512, 19.5155430, -60.9019775, 60.9047737
20: -35.4219284, 21.8182106, -35.4609604, 21.8497581, -57.2716827, 57.2791710
21: -49.2053719, 25.4582863, -49.2686157, 25.5068264, -74.7121964, 74.7268982
22: -50.9541969, 30.0072842, -51.0539207, 30.1147614, -81.0689545, 81.0612030
23: -39.1634941, 26.5992012, -39.2375870, 26.6550159, -65.8185120, 65.8367920
24: -45.2154579, 22.8342056, -45.3123398, 22.8928757, -68.1083374, 68.1465454
25: -38.5025406, 30.9809818, -38.5932236, 31.0805569, -69.5830994, 69.5742035
26: -59.0563698, 37.5599747, -59.1709099, 37.7002869, -96.7566528, 96.7308807
27: -49.4107590, 27.3830566, -49.4618797, 27.3984451, -76.8092041, 76.8449249
28: -37.8677826, 28.8281403, -37.9369812, 28.8882790, -66.7560577, 66.7651215
29: -55.4337120, 34.3235435, -55.5460892, 34.4088860, -89.8425980, 89.8696289
30: -47.7725449, 27.2328606, -47.8673744, 27.2901878, -75.0627289, 75.1002350
31: -49.0315819, 24.0076714, -49.1103859, 24.0474854, -73.0790710, 73.1180573
32: -49.1784325, 27.4757233, -49.2144432, 27.4967651, -76.6751938, 76.6901703
33: -71.8791351, 43.9780807, -71.9676971, 44.0841980, -115.9633331, 115.9457703
34: -60.9104233, 30.0039139, -60.9908371, 30.0952873, -91.0057068, 90.9947510
35: -57.1844292, 34.6630745, -57.2795868, 34.7465210, -91.9309540, 91.9426575
36: -57.2830467, 33.8969116, -57.3537407, 33.9944077, -91.2774506, 91.2506485
37: -85.2246399, 33.0039864, -85.3620605, 33.1369133, -118.3615570, 118.3660355
38: -69.1253052, 41.0039635, -69.1670227, 41.0392609, -110.1645508, 110.1709900
39: -85.0648727, 40.7467422, -85.1286774, 40.8108864, -125.8757629, 125.8754196
40: -75.2217407, 29.9927025, -75.2738800, 30.0426979, -105.2644348, 105.2665863
41: -54.3214188, 25.9177704, -54.3865852, 25.9954033, -80.3168182, 80.3043518
42: -38.9512062, 29.4150162, -38.9822540, 29.4702835, -68.4214935, 68.3972702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 649

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6928939, upper bound: 39.0026628
time: 78.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7764865, upper bound: 39.0026634
time: 88.19 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 169.27 seconds
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 169.27
Output dim: 2, lower bound: -38.6773763, upper bound: 38.9934962
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 169.27
Output dim: 2, lower bound: -38.7617156, upper bound: 38.9935206
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 169.27
Output dim: 2, lower bound: -38.6928939, upper bound: 39.0026628
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 169.27
Output dim: 2, lower bound: -38.7764865, upper bound: 39.0026634

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -53.2478867, 42.8959846, -53.1545868, 42.6745453, -95.9224319, 96.0505676
1: -31.5788784, 35.9629173, -31.5082111, 35.7982864, -67.3771591, 67.4711304
2: -30.3555546, 35.5270081, -30.2855816, 35.2627220, -65.6182709, 65.8125916
3: -33.9029427, 41.5164642, -33.8643417, 41.2580719, -75.1610107, 75.3808060
4: -40.0114441, 38.8209763, -39.9277573, 38.5262680, -78.5377121, 78.7487335
5: -36.8519592, 41.2924767, -36.8143692, 40.9866333, -77.8385925, 78.1068420
6: -55.9313011, 22.4205666, -55.8731079, 22.3322144, -78.2635117, 78.2936707
7: -42.8307800, 40.0256882, -42.7126617, 39.6564484, -82.4872284, 82.7383499
8: -39.2640266, 45.3628044, -39.2235298, 45.0760498, -84.3400726, 84.5863266
9: -34.1747894, 37.4640350, -34.0273972, 37.3020020, -71.4767914, 71.4914322
10: -55.1506577, 52.3698959, -54.8760300, 52.1486893, -107.2993469, 107.2459259
11: -56.4572296, 39.6801033, -56.3950272, 39.6641846, -96.1214066, 96.0751343
12: -58.9989471, 43.9719620, -58.2576599, 43.7151299, -102.7140808, 102.2296219
13: -48.7186890, 49.5716171, -48.4113464, 49.3497810, -98.0684662, 97.9829636
14: -81.2893143, 43.3500824, -80.7534943, 43.1799469, -124.4692612, 124.1035767
15: -40.2651978, 36.3520546, -40.0679550, 36.2990112, -76.5642090, 76.4200134
16: -58.2616768, 40.7106323, -58.0852776, 40.4765930, -98.7382660, 98.7959061
17: -85.0613708, 62.5178871, -84.3994904, 62.2601585, -147.3215332, 146.9173737
18: -48.9456482, 29.1110210, -48.7511215, 29.0391006, -77.9847488, 77.8621368
19: -41.3548241, 19.4461842, -41.2595444, 19.4555931, -60.8104172, 60.7057228
20: -35.3765488, 21.8079205, -35.1949577, 21.7400074, -57.1165543, 57.0028763
21: -49.1630592, 25.4475174, -49.0149422, 25.4182453, -74.5812988, 74.4624557
22: -50.8259850, 29.9895058, -50.3753319, 29.8677006, -80.6936798, 80.3648376
23: -39.1248016, 26.5910912, -39.0001450, 26.5941086, -65.7189026, 65.5912323
24: -45.1821594, 22.8229580, -45.0823669, 22.7546616, -67.9368210, 67.9053192
25: -38.4429588, 30.9595947, -38.3202171, 30.9125118, -69.3554688, 69.2798080
26: -58.9100456, 37.5453796, -58.2337914, 37.3726578, -96.2826996, 95.7791748
27: -49.3743553, 27.3600674, -49.2101364, 27.2358322, -76.6101837, 76.5702057
28: -37.8371048, 28.8170567, -37.7322159, 28.7976379, -66.6347427, 66.5492706
29: -55.3011780, 34.3106613, -54.8747864, 34.1643639, -89.4655457, 89.1854477
30: -47.7381439, 27.2182655, -47.6746597, 27.1640472, -74.9021912, 74.8929214
31: -48.9988251, 23.9862061, -48.8648148, 23.8271713, -72.8259964, 72.8510208
32: -49.1389732, 27.4599266, -48.9515724, 27.3432484, -76.4822235, 76.4114914
33: -71.8542023, 43.9180450, -71.7264633, 43.7259674, -115.5801620, 115.6445084
34: -60.8914108, 29.9746933, -60.8099976, 29.9508533, -90.8422546, 90.7846909
35: -57.1629219, 34.6376152, -57.0992584, 34.6188889, -91.7818146, 91.7368698
36: -57.2211342, 33.8805313, -56.9633408, 33.7998886, -91.0210190, 90.8438721
37: -85.1806641, 32.9781799, -85.0785980, 32.9968033, -118.1774673, 118.0567780
38: -69.0829773, 40.9808273, -68.8755493, 40.8687363, -109.9517136, 109.8563766
39: -85.0369415, 40.6804314, -84.9084396, 40.4825134, -125.5194550, 125.5888672
40: -75.1918411, 29.9147263, -75.0280838, 29.6398354, -104.8316803, 104.9428101
41: -54.2972107, 25.8861618, -54.2244873, 25.8364639, -80.1336746, 80.1106491
42: -38.9152908, 29.3975983, -38.7387238, 29.2979908, -68.2132797, 68.1363220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=207, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.4848207, upper bound: 38.9821356
time: 72.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.4848207, upper bound: 38.9924878
time: 78.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -53.2773743, 42.9677429, -53.2830315, 42.8775330, -96.1549072, 96.2507782
1: -31.5954514, 36.0153198, -31.5992756, 35.9483719, -67.5438232, 67.6145935
2: -30.3689213, 35.5868759, -30.3690910, 35.4330826, -65.8020020, 65.9559631
3: -33.9130211, 41.5787773, -33.9219856, 41.4355812, -75.3485870, 75.5007629
4: -40.0262604, 38.8990631, -40.0192604, 38.7476349, -78.7738953, 78.9183197
5: -36.8631821, 41.3563766, -36.8662033, 41.1689987, -78.0321808, 78.2225800
6: -55.9556694, 22.4415512, -55.9279060, 22.3957329, -78.3514023, 78.3694458
7: -42.8525391, 40.1116867, -42.8298912, 39.8987274, -82.7512665, 82.9415741
8: -39.2780075, 45.4485512, -39.3225784, 45.3229904, -84.6009979, 84.7711334
9: -34.1898499, 37.5027466, -34.0735016, 37.4035721, -71.5934219, 71.5762482
10: -55.1916008, 52.3990097, -54.9890594, 52.2665062, -107.4581070, 107.3880692
11: -56.5015144, 39.6920319, -56.4877129, 39.6992989, -96.2008057, 96.1797485
12: -59.1421280, 43.9915619, -58.6637802, 43.8827972, -103.0249176, 102.6553421
13: -48.7793579, 49.6041565, -48.5817947, 49.4779701, -98.2573242, 98.1859436
14: -81.4109192, 43.3640938, -81.1090393, 43.3045120, -124.7154236, 124.4731216
15: -40.3093414, 36.3785744, -40.1702576, 36.3146210, -76.6239624, 76.5488281
16: -58.2991562, 40.7920837, -58.1974716, 40.6850128, -98.9841614, 98.9895554
17: -85.2129364, 62.5384674, -84.8390045, 62.4248505, -147.6377869, 147.3774719
18: -48.9925232, 29.1237259, -48.8918953, 29.0869179, -78.0794373, 78.0156250
19: -41.3857307, 19.4531498, -41.3317032, 19.4659061, -60.8516388, 60.7848511
20: -35.4211884, 21.8179684, -35.3251038, 21.7896214, -57.2108078, 57.1430664
21: -49.2044754, 25.4577522, -49.1304779, 25.4539986, -74.6584778, 74.5882263
22: -50.9527435, 30.0061264, -50.7370186, 29.9781151, -80.9308624, 80.7431335
23: -39.1629791, 26.5988579, -39.1157608, 26.6205635, -65.7835388, 65.7146149
24: -45.2146835, 22.8338356, -45.1831169, 22.7830086, -67.9976807, 68.0169525
25: -38.5008774, 30.9799843, -38.4824257, 30.9997578, -69.5006332, 69.4624100
26: -59.0550766, 37.5592194, -58.6590958, 37.5055733, -96.5606537, 96.2183151
27: -49.4098587, 27.3821011, -49.3083115, 27.2914696, -76.7013245, 76.6904068
28: -37.8673172, 28.8277779, -37.8255386, 28.8399963, -66.7073059, 66.6533203
29: -55.4325562, 34.3228073, -55.2566376, 34.2806244, -89.7131805, 89.5794449
30: -47.7716522, 27.2324238, -47.7716217, 27.2264576, -74.9981079, 75.0040436
31: -49.0308113, 24.0056686, -48.9415321, 23.8732910, -72.9040985, 72.9471970
32: -49.1776199, 27.4752922, -49.0567589, 27.4033184, -76.5809326, 76.5320511
33: -71.8785248, 43.9772263, -71.8330536, 43.8944092, -115.7729340, 115.8102722
34: -60.9100380, 30.0031776, -60.8856659, 30.0308933, -90.9409180, 90.8888397
35: -57.1839752, 34.6624069, -57.1767921, 34.6809464, -91.8649139, 91.8392029
36: -57.2822914, 33.8963051, -57.1431465, 33.8866997, -91.1689911, 91.0394516
37: -85.2235489, 33.0019264, -85.2085876, 33.0503464, -118.2738953, 118.2105103
38: -69.1238403, 41.0033112, -68.9987183, 40.9367409, -110.0605774, 110.0020294
39: -85.0639191, 40.7461853, -85.0134277, 40.6585617, -125.7224808, 125.7596130
40: -75.2203827, 29.9910965, -75.1434860, 29.8406868, -105.0610657, 105.1345825
41: -54.3204765, 25.9158096, -54.2815628, 25.9108772, -80.2313538, 80.1973724
42: -38.9504700, 29.4145279, -38.8321190, 29.3696747, -68.3201447, 68.2466431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=207, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6876247, upper bound: 38.9821449
time: 78.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7607074, upper bound: 38.9925120
time: 68.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -53.2478867, 42.8959846, -53.2894630, 42.8316422, -96.0795288, 96.1854477
1: -31.5788784, 35.9629173, -31.6018772, 35.9216232, -67.5004959, 67.5647964
2: -30.3555546, 35.5270081, -30.3857040, 35.4838181, -65.8393707, 65.9127121
3: -33.9029427, 41.5164642, -33.9535675, 41.4665337, -75.3694763, 75.4700317
4: -40.0114441, 38.8209763, -40.0194778, 38.7377815, -78.7492218, 78.8404541
5: -36.8519592, 41.2924767, -36.9076729, 41.2432785, -78.0952301, 78.2001495
6: -55.9313011, 22.4205666, -55.9314270, 22.4047852, -78.3360748, 78.3519897
7: -42.8307800, 40.0256882, -42.8684578, 39.9617462, -82.7925262, 82.8941498
8: -39.2640266, 45.3628044, -39.3214989, 45.3173523, -84.5813751, 84.6842957
9: -34.1747894, 37.4640350, -34.1859589, 37.4265366, -71.6013260, 71.6499939
10: -55.1506577, 52.3698959, -55.1318283, 52.3149796, -107.4656372, 107.5017242
11: -56.4572296, 39.6801033, -56.4737053, 39.7132759, -96.1705017, 96.1538086
12: -58.9989471, 43.9719620, -58.7944450, 43.9274826, -102.9264297, 102.7664032
13: -48.7186890, 49.5716171, -48.6346321, 49.5222740, -98.2409515, 98.2062531
14: -81.2893143, 43.3500824, -81.2099152, 43.3332367, -124.6225510, 124.5599899
15: -40.2651978, 36.3520546, -40.2310448, 36.3948135, -76.6600113, 76.5830994
16: -58.2616768, 40.7106323, -58.2462349, 40.5872650, -98.8489380, 98.9568634
17: -85.0613708, 62.5178871, -84.8496017, 62.4236221, -147.4849854, 147.3674774
18: -48.9456482, 29.1110210, -48.9126968, 29.1036720, -78.0493164, 78.0237122
19: -41.3548241, 19.4461842, -41.3714447, 19.4890690, -60.8438911, 60.8176270
20: -35.3765488, 21.8079205, -35.3233528, 21.7977524, -57.1743011, 57.1312714
21: -49.1630592, 25.4475174, -49.1443367, 25.4659042, -74.6289597, 74.5918503
22: -50.8259850, 29.9895058, -50.6790047, 29.9946175, -80.8206024, 80.6685028
23: -39.1248016, 26.5910912, -39.1168327, 26.6247902, -65.7495880, 65.7079163
24: -45.1821594, 22.8229580, -45.2032852, 22.8606949, -68.0428543, 68.0262375
25: -38.4429588, 30.9595947, -38.4168243, 30.9850426, -69.4279938, 69.3764191
26: -58.9100456, 37.5453796, -58.7333603, 37.5600624, -96.4701080, 96.2787323
27: -49.3743553, 27.3600674, -49.3547668, 27.3336773, -76.7080307, 76.7148361
28: -37.8371048, 28.8170567, -37.8390503, 28.8420372, -66.6791382, 66.6561050
29: -55.3011780, 34.3106613, -55.1544685, 34.2863731, -89.5875549, 89.4651337
30: -47.7381439, 27.2182655, -47.7617531, 27.2237968, -74.9619370, 74.9800110
31: -48.9988251, 23.9862061, -49.0252151, 23.9828873, -72.9817123, 73.0114212
32: -49.1389732, 27.4599266, -49.1011925, 27.4313507, -76.5703278, 76.5611115
33: -71.8542023, 43.9180450, -71.8551941, 43.9066315, -115.7608337, 115.7732315
34: -60.8914108, 29.9746933, -60.9112854, 30.0066319, -90.8980408, 90.8859787
35: -57.1629219, 34.6376152, -57.1977005, 34.6737900, -91.8367157, 91.8353119
36: -57.2211342, 33.8805313, -57.1666946, 33.9001312, -91.1212540, 91.0472260
37: -85.1806641, 32.9781799, -85.2224808, 33.0655060, -118.2461700, 118.2006607
38: -69.0829773, 40.9808273, -69.0288391, 40.9639854, -110.0469666, 110.0096664
39: -85.0369415, 40.6804314, -85.0145950, 40.6231232, -125.6600647, 125.6950226
40: -75.1918411, 29.9147263, -75.1461792, 29.8282547, -105.0200882, 105.0609055
41: -54.2972107, 25.8861618, -54.3207016, 25.9033222, -80.2005234, 80.2068634
42: -38.9152908, 29.3975983, -38.8806419, 29.3942947, -68.3095856, 68.2782364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=207, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6184896, upper bound: 38.9911897
time: 88.17 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6918848, upper bound: 38.6982973
time: 90.20 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -53.2773743, 42.9677429, -53.4191093, 43.0355415, -96.3129120, 96.3868561
1: -31.5954514, 36.0153198, -31.6934509, 36.0723991, -67.6678467, 67.7087631
2: -30.3689213, 35.5868759, -30.4694290, 35.6545639, -66.0234833, 66.0562973
3: -33.9130211, 41.5787773, -34.0119743, 41.6447830, -75.5578003, 75.5907516
4: -40.0262604, 38.8990631, -40.1115494, 38.9600143, -78.9862747, 79.0106049
5: -36.8631821, 41.3563766, -36.9601440, 41.4262924, -78.2894669, 78.3164978
6: -55.9556694, 22.4415512, -55.9882927, 22.4684982, -78.4241638, 78.4298401
7: -42.8525391, 40.1116867, -42.9861259, 40.2044220, -83.0569611, 83.0978088
8: -39.2780075, 45.4485512, -39.4211769, 45.5649719, -84.8429794, 84.8697281
9: -34.1898499, 37.5027466, -34.2329292, 37.5340958, -71.7239380, 71.7356720
10: -55.1916008, 52.3990097, -55.2463608, 52.4323311, -107.6239319, 107.6453705
11: -56.5015144, 39.6920319, -56.5685425, 39.7490196, -96.2505341, 96.2605743
12: -59.1421280, 43.9915619, -59.2016029, 44.0954933, -103.2376175, 103.1931610
13: -48.7793579, 49.6041565, -48.8060112, 49.6525307, -98.4318848, 98.4101715
14: -81.4109192, 43.3640938, -81.5667725, 43.4575310, -124.8684387, 124.9308624
15: -40.3093414, 36.3785744, -40.3384171, 36.4123840, -76.7217255, 76.7169952
16: -58.2991562, 40.7920837, -58.3612289, 40.8043594, -99.1035156, 99.1532974
17: -85.2129364, 62.5384674, -85.2899017, 62.5884132, -147.8013458, 147.8283691
18: -48.9925232, 29.1237259, -49.0573769, 29.1522484, -78.1447754, 78.1811066
19: -41.3857307, 19.4531498, -41.4437866, 19.5006485, -60.8863792, 60.8969345
20: -35.4211884, 21.8179684, -35.4546967, 21.8476582, -57.2688446, 57.2726593
21: -49.2044754, 25.4577522, -49.2611160, 25.5019932, -74.7064667, 74.7188644
22: -50.9527435, 30.0061264, -51.0416908, 30.1048412, -81.0575867, 81.0478134
23: -39.1629791, 26.5988579, -39.2329330, 26.6520233, -65.8149948, 65.8317871
24: -45.2146835, 22.8338356, -45.3061066, 22.8896980, -68.1043854, 68.1399384
25: -38.5008774, 30.9799843, -38.5801277, 31.0726032, -69.5734787, 69.5601120
26: -59.0550766, 37.5592194, -59.1596680, 37.6933899, -96.7484665, 96.7188873
27: -49.4098587, 27.3821011, -49.4547310, 27.3899574, -76.7998047, 76.8368301
28: -37.8673172, 28.8277779, -37.9330215, 28.8851337, -66.7524490, 66.7607956
29: -55.4325562, 34.3228073, -55.5368233, 34.4025612, -89.8350983, 89.8596344
30: -47.7716522, 27.2324238, -47.8599777, 27.2865639, -75.0582123, 75.0923996
31: -49.0308113, 24.0056686, -49.1034698, 24.0299530, -73.0607605, 73.1091309
32: -49.1776199, 27.4752922, -49.2078590, 27.4931793, -76.6707916, 76.6831512
33: -71.8785248, 43.9772263, -71.9625549, 44.0765495, -115.9550781, 115.9397736
34: -60.9100380, 30.0031776, -60.9875755, 30.0889874, -90.9990234, 90.9907455
35: -57.1839752, 34.6624069, -57.2757111, 34.7405586, -91.9245300, 91.9381104
36: -57.2822914, 33.8963051, -57.3482018, 33.9888649, -91.2711563, 91.2444916
37: -85.2235489, 33.0019264, -85.3527451, 33.1221504, -118.3457031, 118.3546753
38: -69.1238403, 41.0033112, -69.1544647, 41.0334702, -110.1573105, 110.1577759
39: -85.0639191, 40.7461853, -85.1204681, 40.8059692, -125.8698883, 125.8666534
40: -75.2203827, 29.9910965, -75.2625580, 30.0295506, -105.2499313, 105.2536545
41: -54.3204765, 25.9158096, -54.3785477, 25.9798260, -80.3003006, 80.2943573
42: -38.9504700, 29.4145279, -38.9756927, 29.4662323, -68.4166946, 68.3902130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=207, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7018965, upper bound: 38.9911897
time: 68.27 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7754712, upper bound: 39.0016523
time: 70.77 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 141.46 seconds
IS_A2_B2_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 141.46
Output dim: 2, lower bound: -38.4848207, upper bound: 38.9821356
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 141.46
Output dim: 2, lower bound: -38.4848207, upper bound: 38.9924878
IS_A2_B2_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 141.46
Output dim: 2, lower bound: -38.6876247, upper bound: 38.9821449
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 141.46
Output dim: 2, lower bound: -38.7607074, upper bound: 38.9925120
IS_A2_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 141.46
Output dim: 2, lower bound: -38.6184896, upper bound: 38.9911897
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 141.46
Output dim: 2, lower bound: -38.6918848, upper bound: 38.6982973
IS_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 141.46
Output dim: 2, lower bound: -38.7018965, upper bound: 38.9911897
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 141.46
Output dim: 2, lower bound: -38.7754712, upper bound: 39.0016523

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -53.0973091, 42.8247452, -53.1119423, 42.6672897, -95.7646027, 95.9366913
1: -31.4490643, 35.8873711, -31.4658051, 35.7912941, -67.2403564, 67.3531723
2: -30.2188663, 35.4459305, -30.2406902, 35.2563019, -65.4751587, 65.6866226
3: -33.7319260, 41.4009705, -33.8072586, 41.2489052, -74.9808197, 75.2082291
4: -39.8587646, 38.7194214, -39.8770294, 38.5170593, -78.3758163, 78.5964508
5: -36.6825409, 41.1799355, -36.7573395, 40.9764290, -77.6589661, 77.9372711
6: -55.8519974, 22.3708515, -55.8552666, 22.3178444, -78.1698380, 78.2261200
7: -42.6642075, 39.9402275, -42.6562920, 39.6487198, -82.3129196, 82.5965195
8: -39.1187897, 45.2738953, -39.1762581, 45.0674934, -84.1862793, 84.4501495
9: -34.0318375, 37.4024811, -33.9831848, 37.2923698, -71.3242035, 71.3856659
10: -55.0450096, 52.2431602, -54.8481522, 52.1102486, -107.1552582, 107.0913086
11: -56.3410606, 39.5607300, -56.3780632, 39.6264076, -95.9674683, 95.9387970
12: -58.8940125, 43.8381653, -58.2457504, 43.6724014, -102.5664139, 102.0839157
13: -48.5705338, 49.4705811, -48.3638535, 49.3339577, -97.9044952, 97.8344345
14: -81.1719437, 43.2364273, -80.7270966, 43.1446266, -124.3165588, 123.9635239
15: -40.1641159, 36.2898331, -40.0355301, 36.2889023, -76.4530182, 76.3253632
16: -58.1017380, 40.6692047, -58.0405769, 40.4667816, -98.5685196, 98.7097778
17: -84.9388428, 62.3899231, -84.3694382, 62.2223930, -147.1612396, 146.7593689
18: -48.8643761, 28.9560280, -48.7359695, 28.9896126, -77.8539886, 77.6920013
19: -41.2538147, 19.3275223, -41.2482529, 19.4155312, -60.6693459, 60.5757713
20: -35.3150177, 21.7047806, -35.1834030, 21.7081375, -57.0231514, 56.8881798
21: -49.0639343, 25.3316002, -49.0002670, 25.3799896, -74.4439240, 74.3318634
22: -50.7227592, 29.8627319, -50.3602142, 29.8267899, -80.5495453, 80.2229385
23: -39.0036469, 26.4347305, -38.9885979, 26.5413818, -65.5450287, 65.4233246
24: -45.0689392, 22.6911888, -45.0701637, 22.7118473, -67.7807846, 67.7613449
25: -38.3474655, 30.8106270, -38.3081818, 30.8640823, -69.2115479, 69.1188049
26: -58.8027954, 37.3522110, -58.2202339, 37.3103714, -96.1131668, 95.5724487
27: -49.2709503, 27.2487125, -49.1961975, 27.1988029, -76.4697571, 76.4449081
28: -37.7426453, 28.6761322, -37.7226562, 28.7516041, -66.4942474, 66.3987885
29: -55.1779594, 34.1900063, -54.8575287, 34.1251831, -89.3031387, 89.0475311
30: -47.6511993, 27.1227417, -47.6623611, 27.1343765, -74.7855759, 74.7851028
31: -48.8644371, 23.8559284, -48.8494720, 23.7840652, -72.6484985, 72.7053986
32: -49.0504036, 27.3859177, -48.9370461, 27.3193913, -76.3697968, 76.3229599
33: -71.7730103, 43.8323517, -71.7075806, 43.7001076, -115.4731064, 115.5399323
34: -60.8014526, 29.8329144, -60.7959099, 29.9038811, -90.7053299, 90.6288223
35: -57.1004906, 34.5457458, -57.0861549, 34.5895309, -91.6900177, 91.6318970
36: -57.1604958, 33.7732468, -56.9521561, 33.7638474, -90.9243469, 90.7253952
37: -85.0212097, 32.8222694, -85.0556946, 32.9458237, -117.9670334, 117.8779602
38: -68.9963226, 40.8363571, -68.8587799, 40.8232498, -109.8195724, 109.6951370
39: -84.9230042, 40.5984001, -84.8848724, 40.4556961, -125.3786926, 125.4832687
40: -75.1004181, 29.8614960, -75.0094147, 29.6233330, -104.7237396, 104.8709030
41: -54.2000809, 25.7927094, -54.2102394, 25.8055534, -80.0056305, 80.0029449
42: -38.8467026, 29.3171730, -38.7258644, 29.2722797, -68.1189804, 68.0430374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=207, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 648

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.4148486, upper bound: 38.9793800
time: 77.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5126131, upper bound: 38.9821362
time: 86.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -53.2390213, 42.8929558, -53.1515388, 42.6735229, -95.9125443, 96.0444946
1: -31.5715675, 35.9603424, -31.5057678, 35.7974243, -67.3689880, 67.4661102
2: -30.3491936, 35.5246391, -30.2834587, 35.2619324, -65.6111298, 65.8080978
3: -33.8948898, 41.5134163, -33.8616562, 41.2570724, -75.1519623, 75.3750763
4: -40.0039673, 38.8173561, -39.9252853, 38.5250397, -78.5290070, 78.7426376
5: -36.8434982, 41.2896843, -36.8115234, 40.9856720, -77.8291702, 78.1011963
6: -55.9153519, 22.4170227, -55.8680191, 22.3309860, -78.2463226, 78.2850418
7: -42.8214569, 40.0230217, -42.7095184, 39.6555557, -82.4770126, 82.7325363
8: -39.2559280, 45.3587227, -39.2207947, 45.0746994, -84.3306274, 84.5795135
9: -34.1674995, 37.4599800, -34.0249252, 37.3006630, -71.4681549, 71.4849091
10: -55.1442795, 52.3625374, -54.8738060, 52.1463203, -107.2905731, 107.2363434
11: -56.4512825, 39.6744690, -56.3929787, 39.6623726, -96.1136398, 96.0674438
12: -58.9945221, 43.9653358, -58.2561989, 43.7128754, -102.7073975, 102.2215347
13: -48.7112274, 49.5660820, -48.4089165, 49.3479729, -98.0592041, 97.9749985
14: -81.2816162, 43.3446198, -80.7508774, 43.1781731, -124.4597778, 124.0954895
15: -40.2587013, 36.3488159, -40.0657883, 36.2979469, -76.5566483, 76.4146042
16: -58.2529182, 40.7073555, -58.0823250, 40.4755058, -98.7284241, 98.7896729
17: -85.0527344, 62.5110931, -84.3966370, 62.2579498, -147.3106842, 146.9077148
18: -48.9403687, 29.1030865, -48.7494011, 29.0364914, -77.9768600, 77.8524857
19: -41.3507576, 19.4409695, -41.2582016, 19.4538708, -60.8046265, 60.6991730
20: -35.3729324, 21.8024521, -35.1937370, 21.7381554, -57.1110878, 56.9961891
21: -49.1571960, 25.4424400, -49.0129776, 25.4165802, -74.5737762, 74.4554138
22: -50.8201828, 29.9838543, -50.3734360, 29.8658390, -80.6860123, 80.3572922
23: -39.1208420, 26.5838604, -38.9987602, 26.5917435, -65.7125778, 65.5826187
24: -45.1775360, 22.8165283, -45.0808144, 22.7525215, -67.9300461, 67.8973389
25: -38.4388390, 30.9518414, -38.3188400, 30.9099159, -69.3487549, 69.2706833
26: -58.9047546, 37.5366745, -58.2320671, 37.3697739, -96.2745285, 95.7687378
27: -49.3679657, 27.3551273, -49.2080193, 27.2342567, -76.6022186, 76.5631485
28: -37.8339157, 28.8102264, -37.7311478, 28.7953339, -66.6292496, 66.5413742
29: -55.2936287, 34.3055801, -54.8723030, 34.1627350, -89.4563599, 89.1778793
30: -47.7327194, 27.2136230, -47.6728401, 27.1625042, -74.8952255, 74.8864594
31: -48.9936523, 23.9800034, -48.8631096, 23.8251228, -72.8187714, 72.8431091
32: -49.1345482, 27.4537582, -48.9501114, 27.3412170, -76.4757614, 76.4038696
33: -71.8454132, 43.9127998, -71.7235870, 43.7242355, -115.5696487, 115.6363831
34: -60.8876610, 29.9670868, -60.8087387, 29.9483509, -90.8360138, 90.7758179
35: -57.1596298, 34.6320953, -57.0981865, 34.6169891, -91.7766190, 91.7302780
36: -57.2174416, 33.8742981, -56.9621277, 33.7978058, -91.0152435, 90.8364258
37: -85.1737747, 32.9710312, -85.0763016, 32.9944534, -118.1682129, 118.0473328
38: -69.0786285, 40.9719238, -68.8741150, 40.8657990, -109.9444275, 109.8460388
39: -85.0315933, 40.6747665, -84.9066544, 40.4806252, -125.5122070, 125.5814209
40: -75.1858521, 29.9115105, -75.0260925, 29.6387596, -104.8246002, 104.9376068
41: -54.2917709, 25.8810215, -54.2226791, 25.8347740, -80.1265411, 80.1036987
42: -38.9105759, 29.3921280, -38.7370911, 29.2958069, -68.2063828, 68.1292191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=207, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 648

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5856317, upper bound: 38.9896978
time: 70.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5856317, upper bound: 38.9896982
time: 292.20 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -53.1268005, 42.8964729, -53.2403641, 42.8702621, -95.9970627, 96.1368408
1: -31.4656506, 35.9398079, -31.5568886, 35.9414062, -67.4070511, 67.4966965
2: -30.2322178, 35.5057983, -30.3241711, 35.4266510, -65.6588669, 65.8299713
3: -33.7420235, 41.4632874, -33.8649216, 41.4263725, -75.1683960, 75.3282089
4: -39.8735809, 38.7974930, -39.9685135, 38.7384415, -78.6120148, 78.7659912
5: -36.6937790, 41.2438202, -36.8091621, 41.1587906, -77.8525696, 78.0529785
6: -55.8763695, 22.3918304, -55.9100533, 22.3813362, -78.2577057, 78.3018799
7: -42.6859589, 40.0262566, -42.7735138, 39.8910103, -82.5769653, 82.7997665
8: -39.1327744, 45.3596344, -39.2752647, 45.3144379, -84.4472122, 84.6348877
9: -34.0469055, 37.4411697, -34.0292969, 37.3940048, -71.4409103, 71.4704666
10: -55.0861130, 52.2722931, -54.9612350, 52.2280045, -107.3141098, 107.2335281
11: -56.3853416, 39.5726509, -56.4707375, 39.6614647, -96.0468063, 96.0433884
12: -59.0372009, 43.8577309, -58.6518860, 43.8400726, -102.8772736, 102.5096130
13: -48.6312370, 49.5030632, -48.5343170, 49.4621239, -98.0933533, 98.0373764
14: -81.2935028, 43.2503662, -81.0826416, 43.2691917, -124.5626907, 124.3330002
15: -40.2082558, 36.3163223, -40.1378708, 36.3043594, -76.5126038, 76.4541931
16: -58.1393204, 40.7506561, -58.1527557, 40.6752625, -98.8145752, 98.9034119
17: -85.0903549, 62.4104691, -84.8089447, 62.3870621, -147.4774170, 147.2194061
18: -48.9112511, 28.9687271, -48.8767624, 29.0373917, -77.9486389, 77.8454895
19: -41.2846985, 19.3344975, -41.3204269, 19.4258499, -60.7105484, 60.6549225
20: -35.3596420, 21.7148151, -35.3135529, 21.7577362, -57.1173706, 57.0283661
21: -49.1053162, 25.3418159, -49.1158142, 25.4157410, -74.5210495, 74.4576263
22: -50.8495407, 29.8793125, -50.7219543, 29.9371891, -80.7867279, 80.6012650
23: -39.0418396, 26.4424973, -39.1042061, 26.5678444, -65.6096802, 65.5467072
24: -45.1014404, 22.7020760, -45.1708870, 22.7402115, -67.8416519, 67.8729553
25: -38.4054260, 30.8310318, -38.4704514, 30.9513302, -69.3567505, 69.3014832
26: -58.9478455, 37.3660622, -58.6455460, 37.4432869, -96.3911285, 96.0116119
27: -49.3064270, 27.2707958, -49.2943420, 27.2544708, -76.5608978, 76.5651398
28: -37.7728348, 28.6867962, -37.8159637, 28.7939415, -66.5667725, 66.5027618
29: -55.3093452, 34.2021637, -55.2393608, 34.2413979, -89.5507431, 89.4415283
30: -47.6847076, 27.1369019, -47.7593231, 27.1967869, -74.8814926, 74.8962173
31: -48.8963814, 23.8754120, -48.9261169, 23.8302155, -72.7265930, 72.8015289
32: -49.0890160, 27.4013195, -49.0422134, 27.3794899, -76.4684982, 76.4435349
33: -71.7973022, 43.8915405, -71.8141174, 43.8685837, -115.6658707, 115.7056580
34: -60.8200607, 29.8614197, -60.8715515, 29.9839172, -90.8039780, 90.7329712
35: -57.1215057, 34.5705338, -57.1636467, 34.6516151, -91.7731171, 91.7341766
36: -57.2216721, 33.7889786, -57.1320419, 33.8506851, -91.0723572, 90.9210205
37: -85.0640488, 32.8460541, -85.1855774, 32.9993591, -118.0634079, 118.0316238
38: -69.0372162, 40.8588562, -68.9819183, 40.8911858, -109.9284058, 109.8407745
39: -84.9499130, 40.6641617, -84.9898148, 40.6317253, -125.5816345, 125.6539764
40: -75.1289215, 29.9378910, -75.1247559, 29.8241863, -104.9530945, 105.0626450
41: -54.2233276, 25.8224545, -54.2672729, 25.8800278, -80.1033478, 80.0897217
42: -38.8818130, 29.3340874, -38.8191986, 29.3439445, -68.2257538, 68.1532745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=207, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 648

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5717272, upper bound: 38.9800429
time: 138.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.4148486, upper bound: 38.9821449
time: 77.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -53.2685127, 42.9647293, -53.2799606, 42.8765488, -96.1450500, 96.2446899
1: -31.5881500, 36.0127487, -31.5968399, 35.9475174, -67.5356674, 67.6095886
2: -30.3625488, 35.5844955, -30.3669567, 35.4323044, -65.7948532, 65.9514465
3: -33.9049911, 41.5757256, -33.9193192, 41.4345474, -75.3395386, 75.4950409
4: -40.0187721, 38.8954315, -40.0167770, 38.7463837, -78.7651520, 78.9122009
5: -36.8547287, 41.3535538, -36.8633690, 41.1680183, -78.0227356, 78.2169189
6: -55.9397240, 22.4379997, -55.9228134, 22.3944817, -78.3342056, 78.3608093
7: -42.8432007, 40.1090469, -42.8267288, 39.8978348, -82.7410355, 82.9357758
8: -39.2699432, 45.4444656, -39.3198318, 45.3216782, -84.5916214, 84.7642975
9: -34.1825752, 37.4987030, -34.0710068, 37.4022408, -71.5848160, 71.5697021
10: -55.1852379, 52.3916740, -54.9868622, 52.2641296, -107.4493713, 107.3785400
11: -56.4955750, 39.6863899, -56.4856567, 39.6974716, -96.1930466, 96.1720428
12: -59.1377106, 43.9849167, -58.6623077, 43.8805161, -103.0182190, 102.6472244
13: -48.7719269, 49.5986099, -48.5793686, 49.4761429, -98.2480698, 98.1779785
14: -81.4031830, 43.3586273, -81.1064148, 43.3027649, -124.7059479, 124.4650421
15: -40.3028488, 36.3753319, -40.1680756, 36.3135452, -76.6163940, 76.5434036
16: -58.2904091, 40.7888069, -58.1944847, 40.6838913, -98.9742889, 98.9832916
17: -85.2042694, 62.5316200, -84.8361740, 62.4226418, -147.6269073, 147.3677979
18: -48.9872475, 29.1157742, -48.8901863, 29.0843029, -78.0715485, 78.0059586
19: -41.3816605, 19.4479122, -41.3303680, 19.4642143, -60.8458710, 60.7782822
20: -35.4175911, 21.8124847, -35.3238831, 21.7877636, -57.2053528, 57.1363640
21: -49.1986237, 25.4526615, -49.1285248, 25.4523182, -74.6509399, 74.5811844
22: -50.9469643, 30.0004673, -50.7351151, 29.9762478, -80.9232025, 80.7355804
23: -39.1590042, 26.5916233, -39.1143723, 26.6182137, -65.7772217, 65.7059937
24: -45.2100639, 22.8273926, -45.1815758, 22.7808685, -67.9909286, 68.0089722
25: -38.4967728, 30.9722691, -38.4810638, 30.9971447, -69.4939117, 69.4533310
26: -59.0497856, 37.5505028, -58.6573410, 37.5026932, -96.5524750, 96.2078400
27: -49.4034576, 27.3771496, -49.3062019, 27.2898884, -76.6933441, 76.6833496
28: -37.8641434, 28.8209381, -37.8244781, 28.8376732, -66.7018127, 66.6454086
29: -55.4249954, 34.3177109, -55.2541428, 34.2789612, -89.7039566, 89.5718536
30: -47.7662315, 27.2277851, -47.7698135, 27.2248955, -74.9911270, 74.9975967
31: -49.0256233, 23.9994602, -48.9398041, 23.8712425, -72.8968658, 72.9392624
32: -49.1731758, 27.4691372, -49.0552711, 27.4013004, -76.5744781, 76.5244064
33: -71.8697205, 43.9719963, -71.8301849, 43.8926544, -115.7623672, 115.8021774
34: -60.9063492, 29.9956284, -60.8844299, 30.0283680, -90.9347153, 90.8800583
35: -57.1806488, 34.6568909, -57.1756783, 34.6790619, -91.8597107, 91.8325653
36: -57.2786446, 33.8899918, -57.1419525, 33.8846283, -91.1632690, 91.0319443
37: -85.2166290, 32.9947662, -85.2062531, 33.0479774, -118.2646027, 118.2010193
38: -69.1194992, 40.9943924, -68.9972839, 40.9337158, -110.0532150, 109.9916687
39: -85.0585938, 40.7405472, -85.0116653, 40.6566620, -125.7152557, 125.7522125
40: -75.2144089, 29.9878521, -75.1414642, 29.8396187, -105.0540314, 105.1293182
41: -54.3150330, 25.9106655, -54.2797508, 25.9091988, -80.2242203, 80.1904144
42: -38.9457512, 29.4090462, -38.8304977, 29.3675156, -68.3132629, 68.2395477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=207, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 648

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6449095, upper bound: 38.9903443
time: 113.29 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6449095, upper bound: 38.9925126
time: 86.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -53.0973091, 42.8247452, -53.2468262, 42.8244438, -95.9217529, 96.0715714
1: -31.4490643, 35.8873711, -31.5594921, 35.9147377, -67.3638000, 67.4468613
2: -30.2188663, 35.4459305, -30.3407879, 35.4773941, -65.6962585, 65.7867203
3: -33.7319260, 41.4009705, -33.8964996, 41.4573708, -75.1893005, 75.2974701
4: -39.8587646, 38.7194214, -39.9687538, 38.7285995, -78.5873642, 78.6881714
5: -36.6825409, 41.1799355, -36.8506470, 41.2330933, -77.9156342, 78.0305786
6: -55.8519974, 22.3708515, -55.9135857, 22.3904037, -78.2424011, 78.2844315
7: -42.6642075, 39.9402275, -42.8121071, 39.9540749, -82.6182861, 82.7523346
8: -39.1187897, 45.2738953, -39.2742157, 45.3088303, -84.4276199, 84.5481110
9: -34.0318375, 37.4024811, -34.1417770, 37.4168930, -71.4487305, 71.5442581
10: -55.0450096, 52.2431602, -55.1039886, 52.2765312, -107.3215332, 107.3471451
11: -56.3410606, 39.5607300, -56.4567757, 39.6754837, -96.0165405, 96.0175018
12: -58.8940125, 43.8381653, -58.7825775, 43.8847542, -102.7787628, 102.6207428
13: -48.5705338, 49.4705811, -48.5871353, 49.5064316, -98.0769653, 98.0577164
14: -81.1719437, 43.2364273, -81.1835556, 43.2978859, -124.4698181, 124.4199829
15: -40.1641159, 36.2898331, -40.1986160, 36.3846588, -76.5487747, 76.4884491
16: -58.1017380, 40.6692047, -58.2015533, 40.5774689, -98.6792068, 98.8707504
17: -84.9388428, 62.3899231, -84.8195496, 62.3857651, -147.3246002, 147.2094727
18: -48.8643761, 28.9560280, -48.8974686, 29.0541878, -77.9185638, 77.8535004
19: -41.2538147, 19.3275223, -41.3601227, 19.4490089, -60.7028122, 60.6876411
20: -35.3150177, 21.7047806, -35.3118210, 21.7658691, -57.0808868, 57.0166016
21: -49.0639343, 25.3316002, -49.1296616, 25.4276505, -74.4915848, 74.4612579
22: -50.7227592, 29.8627319, -50.6639404, 29.9537067, -80.6764603, 80.5266647
23: -39.0036469, 26.4347305, -39.1052437, 26.5720711, -65.5757065, 65.5399780
24: -45.0689392, 22.6911888, -45.1910439, 22.8178940, -67.8868256, 67.8822327
25: -38.3474655, 30.8106270, -38.4047699, 30.9366055, -69.2840729, 69.2153931
26: -58.8027954, 37.3522110, -58.7198563, 37.4978371, -96.3006287, 96.0720673
27: -49.2709503, 27.2487125, -49.3407936, 27.2966537, -76.5675964, 76.5895004
28: -37.7426453, 28.6761322, -37.8294640, 28.7960052, -66.5386505, 66.5056000
29: -55.1779594, 34.1900063, -55.1371918, 34.2471657, -89.4251251, 89.3271942
30: -47.6511993, 27.1227417, -47.7494087, 27.1941109, -74.8453064, 74.8721466
31: -48.8644371, 23.8559284, -49.0098190, 23.9398212, -72.8042526, 72.8657455
32: -49.0504036, 27.3859177, -49.0866394, 27.4075203, -76.4579163, 76.4725571
33: -71.7730103, 43.8323517, -71.8362732, 43.8808250, -115.6538239, 115.6686249
34: -60.8014526, 29.8329144, -60.8971214, 29.9596691, -90.7611160, 90.7300339
35: -57.1004906, 34.5457458, -57.1845932, 34.6444130, -91.7449036, 91.7303391
36: -57.1604958, 33.7732468, -57.1555672, 33.8641243, -91.0246124, 90.9288025
37: -85.0212097, 32.8222694, -85.1994858, 33.0145187, -118.0357285, 118.0217590
38: -68.9963226, 40.8363571, -69.0120697, 40.9184570, -109.9147720, 109.8484268
39: -84.9230042, 40.5984001, -84.9909821, 40.5962906, -125.5192871, 125.5893860
40: -75.1004181, 29.8614960, -75.1275482, 29.8117523, -104.9121704, 104.9890442
41: -54.2000809, 25.7927094, -54.3064346, 25.8724613, -80.0725403, 80.0991440
42: -38.8467026, 29.3171730, -38.8677826, 29.3685608, -68.2152557, 68.1849518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=207, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 648

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5312020, upper bound: 38.9896817
time: 83.26 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5312020, upper bound: 38.9911905
time: 86.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -53.1268005, 42.8964729, -53.3764725, 43.0283203, -96.1551208, 96.2729416
1: -31.4656506, 35.9398079, -31.6510811, 36.0655289, -67.5311813, 67.5908890
2: -30.2322178, 35.5057983, -30.4245243, 35.6481400, -65.8803558, 65.9303207
3: -33.7420235, 41.4632874, -33.9549141, 41.6356392, -75.3776627, 75.4181976
4: -39.8735809, 38.7974930, -40.0608139, 38.9508362, -78.8244095, 78.8583069
5: -36.6937790, 41.2438202, -36.9031296, 41.4161224, -78.1099014, 78.1469498
6: -55.8763695, 22.3918304, -55.9704208, 22.4540939, -78.3304520, 78.3622513
7: -42.6859589, 40.0262566, -42.9297714, 40.1967239, -82.8826828, 82.9560242
8: -39.1327744, 45.3596344, -39.3739052, 45.5564346, -84.6892090, 84.7335358
9: -34.0469055, 37.4411697, -34.1887360, 37.5245361, -71.5714340, 71.6299057
10: -55.0861130, 52.2722931, -55.2186050, 52.3938026, -107.4799042, 107.4908981
11: -56.3853416, 39.5726509, -56.5516281, 39.7111931, -96.0965347, 96.1242752
12: -59.0372009, 43.8577309, -59.1897278, 44.0527954, -103.0899963, 103.0474548
13: -48.6312370, 49.5030632, -48.7585640, 49.6366348, -98.2678680, 98.2616272
14: -81.2935028, 43.2503662, -81.5404510, 43.4221382, -124.7156372, 124.7908173
15: -40.2082558, 36.3163223, -40.3060150, 36.4020805, -76.6103363, 76.6223373
16: -58.1393204, 40.7506561, -58.3165207, 40.7946510, -98.9339752, 99.0671768
17: -85.0903549, 62.4104691, -85.2599106, 62.5506020, -147.6409607, 147.6703796
18: -48.9112511, 28.9687271, -49.0421753, 29.1027145, -78.0139618, 78.0109024
19: -41.2846985, 19.3344975, -41.4324646, 19.4605885, -60.7452850, 60.7669601
20: -35.3596420, 21.7148151, -35.4431763, 21.8157864, -57.1754303, 57.1579895
21: -49.1053162, 25.3418159, -49.2464371, 25.4637260, -74.5690308, 74.5882568
22: -50.8495407, 29.8793125, -51.0266647, 30.0639038, -80.9134445, 80.9059753
23: -39.0418396, 26.4424973, -39.2213516, 26.5992813, -65.6411209, 65.6638489
24: -45.1014404, 22.7020760, -45.2938690, 22.8469124, -67.9483490, 67.9959412
25: -38.4054260, 30.8310318, -38.5681152, 31.0241661, -69.4295807, 69.3991394
26: -58.9478455, 37.3660622, -59.1462326, 37.6311378, -96.5789795, 96.5122910
27: -49.3064270, 27.2707958, -49.4407234, 27.3529663, -76.6593933, 76.7115173
28: -37.7728348, 28.6867962, -37.9234390, 28.8390942, -66.6119232, 66.6102295
29: -55.3093452, 34.2021637, -55.5195999, 34.3633308, -89.6726685, 89.7217636
30: -47.6847076, 27.1369019, -47.8476486, 27.2568932, -74.9415894, 74.9845428
31: -48.8963814, 23.8754120, -49.0880280, 23.9868965, -72.8832703, 72.9634399
32: -49.0890160, 27.4013195, -49.1933250, 27.4693089, -76.5583267, 76.5946426
33: -71.7973022, 43.8915405, -71.9436035, 44.0507927, -115.8480988, 115.8351440
34: -60.8200607, 29.8614197, -60.9733810, 30.0420399, -90.8620987, 90.8347931
35: -57.1215057, 34.5705338, -57.2625656, 34.7111664, -91.8326645, 91.8330994
36: -57.2216721, 33.7889786, -57.3371391, 33.9528427, -91.1744995, 91.1261139
37: -85.0640488, 32.8460541, -85.3297272, 33.0711823, -118.1352310, 118.1757812
38: -69.0372162, 40.8588562, -69.1376801, 40.9879417, -110.0251465, 109.9965363
39: -84.9499130, 40.6641617, -85.0968323, 40.7791367, -125.7290497, 125.7609863
40: -75.1289215, 29.9378910, -75.2438660, 30.0130386, -105.1419525, 105.1817551
41: -54.2233276, 25.8224545, -54.3642311, 25.9490089, -80.1723328, 80.1866837
42: -38.8818130, 29.3340874, -38.9626999, 29.4404755, -68.3222885, 68.2967834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=207, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 648

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5882087, upper bound: 38.9896822
time: 75.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.4148486, upper bound: 38.8605300
time: 95.27 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -53.2685127, 42.9647293, -53.4160614, 43.0345383, -96.3030548, 96.3807907
1: -31.5881500, 36.0127487, -31.6910000, 36.0715561, -67.6597061, 67.7037430
2: -30.3625488, 35.5844955, -30.4673080, 35.6537857, -66.0163345, 66.0518036
3: -33.9049911, 41.5757256, -34.0093079, 41.6438103, -75.5487976, 75.5850296
4: -40.0187721, 38.8954315, -40.1090736, 38.9587669, -78.9775391, 79.0045013
5: -36.8547287, 41.3535538, -36.9573135, 41.4253464, -78.2800751, 78.3108597
6: -55.9397240, 22.4379997, -55.9831963, 22.4672394, -78.4069519, 78.4211884
7: -42.8432007, 40.1090469, -42.9829865, 40.2035599, -83.0467606, 83.0920334
8: -39.2699432, 45.4444656, -39.4184685, 45.5636215, -84.8335571, 84.8629303
9: -34.1825752, 37.4987030, -34.2304535, 37.5327644, -71.7153397, 71.7291565
10: -55.1852379, 52.3916740, -55.2441254, 52.4299622, -107.6152039, 107.6357956
11: -56.4955750, 39.6863899, -56.5665359, 39.7471886, -96.2427673, 96.2529297
12: -59.1377106, 43.9849167, -59.2001381, 44.0932617, -103.2309570, 103.1850586
13: -48.7719269, 49.5986099, -48.8035889, 49.6507187, -98.4226379, 98.4021988
14: -81.4031830, 43.3586273, -81.5641708, 43.4557800, -124.8589478, 124.9227982
15: -40.3028488, 36.3753319, -40.3362274, 36.4113235, -76.7141724, 76.7115555
16: -58.2904091, 40.7888069, -58.3582497, 40.8032608, -99.0936737, 99.1470566
17: -85.2042694, 62.5316200, -85.2870483, 62.5861969, -147.7904663, 147.8186646
18: -48.9872475, 29.1157742, -49.0556450, 29.1496124, -78.1368561, 78.1714172
19: -41.3816605, 19.4479122, -41.4424591, 19.4989471, -60.8806076, 60.8903732
20: -35.4175911, 21.8124847, -35.4534760, 21.8458118, -57.2634048, 57.2659531
21: -49.1986237, 25.4526615, -49.2591400, 25.5003471, -74.6989746, 74.7117996
22: -50.9469643, 30.0004673, -51.0397758, 30.1029530, -81.0499191, 81.0402451
23: -39.1590042, 26.5916233, -39.2315216, 26.6496658, -65.8086700, 65.8231430
24: -45.2100639, 22.8273926, -45.3045349, 22.8875618, -68.0976257, 68.1319275
25: -38.4967728, 30.9722691, -38.5787506, 31.0699902, -69.5667648, 69.5510178
26: -59.0497856, 37.5505028, -59.1579590, 37.6905212, -96.7402954, 96.7084656
27: -49.4034576, 27.3771496, -49.4526291, 27.3883667, -76.7918091, 76.8297729
28: -37.8641434, 28.8209381, -37.9319534, 28.8828564, -66.7469940, 66.7528839
29: -55.4249954, 34.3177109, -55.5343475, 34.4009209, -89.8259125, 89.8520584
30: -47.7662315, 27.2277851, -47.8581696, 27.2850037, -75.0512238, 75.0859528
31: -49.0256233, 23.9994602, -49.1017570, 24.0278950, -73.0535126, 73.1012115
32: -49.1731758, 27.4691372, -49.2063751, 27.4911594, -76.6643372, 76.6755142
33: -71.8697205, 43.9719963, -71.9596710, 44.0747795, -115.9444962, 115.9316711
34: -60.9063492, 29.9956284, -60.9863319, 30.0864944, -90.9928207, 90.9819641
35: -57.1806488, 34.6568909, -57.2746429, 34.7386589, -91.9193039, 91.9315186
36: -57.2786446, 33.8899918, -57.3469963, 33.9867706, -91.2654114, 91.2369843
37: -85.2166290, 32.9947662, -85.3503571, 33.1198235, -118.3364563, 118.3451233
38: -69.1194992, 40.9943924, -69.1529999, 41.0304794, -110.1499786, 110.1473923
39: -85.0585938, 40.7405472, -85.1186752, 40.8041153, -125.8627090, 125.8592224
40: -75.2144089, 29.9878521, -75.2605667, 30.0284805, -105.2428894, 105.2484207
41: -54.3150330, 25.9106655, -54.3767242, 25.9781399, -80.2931595, 80.2873840
42: -38.9457512, 29.4090462, -38.9740601, 29.4640484, -68.4097977, 68.3831024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=207, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 648

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6619142, upper bound: 39.0001417
time: 76.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6619145, upper bound: 39.0016522
time: 67.05 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 146.20 seconds
IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.4148486, upper bound: 38.9793800
IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.5126131, upper bound: 38.9821362
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.5856317, upper bound: 38.9896978
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.5856317, upper bound: 38.9896982
IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.5717272, upper bound: 38.9800429
IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.4148486, upper bound: 38.9821449
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.6449095, upper bound: 38.9903443
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.6449095, upper bound: 38.9925126
IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.5312020, upper bound: 38.9896817
IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.5312020, upper bound: 38.9911905
IS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.5882087, upper bound: 38.9896822
IS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.4148486, upper bound: 38.8605300
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.6619142, upper bound: 39.0001417
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 146.20
Output dim: 2, lower bound: -38.6619145, upper bound: 39.0016522

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -53.0973091, 42.8247452, -53.0104141, 42.4930191, -95.5903168, 95.8351593
1: -31.4490643, 35.8873711, -31.3856354, 35.6433563, -67.0924225, 67.2730026
2: -30.2188663, 35.4459305, -30.1595097, 35.0778503, -65.2967148, 65.6054382
3: -33.7319260, 41.4009705, -33.7484741, 41.0836678, -74.8155975, 75.1494446
4: -39.8587646, 38.7194214, -39.7938805, 38.3295898, -78.1883545, 78.5132980
5: -36.6825409, 41.1799355, -36.6871147, 40.7736626, -77.4562073, 77.8670502
6: -55.8519974, 22.3708515, -55.8287430, 22.2209854, -78.0729828, 78.1995850
7: -42.6642075, 39.9402275, -42.5238113, 39.3704910, -82.0346985, 82.4640350
8: -39.1187897, 45.2738953, -39.0858688, 44.8500595, -83.9688492, 84.3597641
9: -34.0318375, 37.4024811, -33.8779373, 37.2095108, -71.2413483, 71.2804184
10: -55.0450096, 52.2431602, -54.6611328, 51.9865494, -107.0315475, 106.9042892
11: -56.3410606, 39.5607300, -56.3398590, 39.5820198, -95.9230804, 95.9005890
12: -58.8940125, 43.8381653, -57.8644028, 43.5297241, -102.4237366, 101.7025681
13: -48.5705338, 49.4705811, -48.1761322, 49.2076721, -97.7781906, 97.6467133
14: -81.1719437, 43.2364273, -80.4025421, 43.0219879, -124.1939316, 123.6389694
15: -40.1641159, 36.2898331, -39.8758812, 36.2458115, -76.4099197, 76.1657104
16: -58.1017380, 40.6692047, -57.9633942, 40.3332977, -98.4350357, 98.6325912
17: -84.9388428, 62.3899231, -84.0693207, 62.0979652, -147.0368042, 146.4592438
18: -48.8643761, 28.9560280, -48.6151581, 28.9486122, -77.8129883, 77.5711823
19: -41.2538147, 19.3275223, -41.1783791, 19.3922424, -60.6460571, 60.5058899
20: -35.3150177, 21.7047806, -35.0973816, 21.6723442, -56.9873581, 56.8021622
21: -49.0639343, 25.3316002, -48.9031143, 25.3397102, -74.4036407, 74.2347107
22: -50.7227592, 29.8627319, -50.0874405, 29.7175694, -80.4403305, 79.9501648
23: -39.0036469, 26.4347305, -38.9050941, 26.5105629, -65.5142059, 65.3398209
24: -45.0689392, 22.6911888, -45.0006561, 22.6676598, -67.7366028, 67.6918488
25: -38.3474655, 30.8106270, -38.1873779, 30.7816048, -69.1290665, 68.9980011
26: -58.8027954, 37.3522110, -57.8486748, 37.1778069, -95.9806061, 95.2008820
27: -49.2709503, 27.2487125, -49.0871925, 27.1071510, -76.3780899, 76.3358994
28: -37.7426453, 28.6761322, -37.6433182, 28.7084713, -66.4511185, 66.3194504
29: -55.1779594, 34.1900063, -54.5869446, 34.0226517, -89.2006073, 88.7769470
30: -47.6511993, 27.1227417, -47.6045876, 27.0884914, -74.7396927, 74.7273254
31: -48.8644371, 23.8559284, -48.7644005, 23.7064323, -72.5708694, 72.6203308
32: -49.0504036, 27.3859177, -48.8397369, 27.2723141, -76.3227158, 76.2256546
33: -71.7730103, 43.8323517, -71.6306000, 43.6057739, -115.3787842, 115.4629517
34: -60.8014526, 29.8329144, -60.7272415, 29.8718929, -90.6733398, 90.5601501
35: -57.1004906, 34.5457458, -57.0302658, 34.5716019, -91.6720886, 91.5760117
36: -57.1604958, 33.7732468, -56.8075905, 33.6980057, -90.8585052, 90.5808334
37: -85.0212097, 32.8222694, -84.9478760, 32.8987122, -117.9199219, 117.7701416
38: -68.9963226, 40.8363571, -68.7675400, 40.7855339, -109.7818604, 109.6038971
39: -84.9230042, 40.5984001, -84.8168564, 40.3972855, -125.3202896, 125.4152527
40: -75.1004181, 29.8614960, -74.9102936, 29.4581223, -104.5585251, 104.7717896
41: -54.2000809, 25.7927094, -54.1471291, 25.7228222, -79.9228973, 79.9398346
42: -38.8467026, 29.3171730, -38.6511383, 29.2092857, -68.0559845, 67.9683075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=206, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=405, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 695

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.4460657, upper bound: 38.9772307
time: 69.31 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.3481789, upper bound: 38.9064174
time: 80.48 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -53.0973091, 42.8247452, -53.1069336, 42.6624603, -95.7597656, 95.9316788
1: -31.4490643, 35.8873711, -31.4608917, 35.7880554, -67.2371216, 67.3482590
2: -30.2188663, 35.4459305, -30.2388725, 35.2525215, -65.4713898, 65.6847992
3: -33.7319260, 41.4009705, -33.8056221, 41.2438583, -74.9757843, 75.2065887
4: -39.8587646, 38.7194214, -39.8745193, 38.5124664, -78.3712311, 78.5939331
5: -36.6825409, 41.1799355, -36.7558403, 40.9710922, -77.6536331, 77.9357758
6: -55.8519974, 22.3708515, -55.8504791, 22.2991714, -78.1511688, 78.2213287
7: -42.6642075, 39.9402275, -42.6487350, 39.6432076, -82.3074036, 82.5889587
8: -39.1187897, 45.2738953, -39.1716843, 45.0622635, -84.1810455, 84.4455795
9: -34.0318375, 37.4024811, -33.9781876, 37.2896271, -71.3214645, 71.3806686
10: -55.0450096, 52.2431602, -54.8421135, 52.1075211, -107.1525269, 107.0852737
11: -56.3410606, 39.5607300, -56.3705330, 39.6108170, -95.9518738, 95.9312592
12: -58.8940125, 43.8381653, -58.2369270, 43.6704521, -102.5644684, 102.0750885
13: -48.5705338, 49.4705811, -48.3554382, 49.3231468, -97.8936768, 97.8260193
14: -81.1719437, 43.2364273, -80.7191010, 43.1410217, -124.3129654, 123.9555283
15: -40.1641159, 36.2898331, -40.0101357, 36.2818298, -76.4459381, 76.2999649
16: -58.1017380, 40.6692047, -58.0328445, 40.4423714, -98.5441132, 98.7020416
17: -84.9388428, 62.3899231, -84.3631668, 62.2180176, -147.1568604, 146.7530823
18: -48.8643761, 28.9560280, -48.7290344, 28.9810543, -77.8454285, 77.6850586
19: -41.2538147, 19.3275223, -41.2448807, 19.4085255, -60.6623383, 60.5723991
20: -35.3150177, 21.7047806, -35.1793060, 21.7067432, -57.0217552, 56.8840866
21: -49.0639343, 25.3316002, -48.9950981, 25.3785744, -74.4425049, 74.3266983
22: -50.7227592, 29.8627319, -50.3470383, 29.8189583, -80.5417175, 80.2097626
23: -39.0036469, 26.4347305, -38.9855423, 26.5395737, -65.5432129, 65.4202728
24: -45.0689392, 22.6911888, -45.0666389, 22.7099037, -67.7788391, 67.7578125
25: -38.3474655, 30.8106270, -38.2995605, 30.8579159, -69.2053833, 69.1101837
26: -58.8027954, 37.3522110, -58.2112770, 37.3059387, -96.1087341, 95.5634918
27: -49.2709503, 27.2487125, -49.1890259, 27.1944866, -76.4654388, 76.4377365
28: -37.7426453, 28.6761322, -37.7202301, 28.7487297, -66.4913788, 66.3963623
29: -55.1779594, 34.1900063, -54.8501663, 34.1200142, -89.2979660, 89.0401764
30: -47.6511993, 27.1227417, -47.6568375, 27.1326237, -74.7838211, 74.7795792
31: -48.8644371, 23.8559284, -48.8445091, 23.7772427, -72.6416702, 72.7004395
32: -49.0504036, 27.3859177, -48.9322319, 27.3176880, -76.3680878, 76.3181458
33: -71.7730103, 43.8323517, -71.7045593, 43.6948814, -115.4678802, 115.5369034
34: -60.8014526, 29.8329144, -60.7932930, 29.9001904, -90.7016449, 90.6262054
35: -57.1004906, 34.5457458, -57.0781021, 34.5854874, -91.6859741, 91.6238480
36: -57.1604958, 33.7732468, -56.9419708, 33.7590714, -90.9195633, 90.7152176
37: -85.0212097, 32.8222694, -85.0496521, 32.9422302, -117.9634399, 117.8719177
38: -68.9963226, 40.8363571, -68.8550034, 40.8205643, -109.8168869, 109.6913528
39: -84.9230042, 40.5984001, -84.8801575, 40.4524765, -125.3754807, 125.4785614
40: -75.1004181, 29.8614960, -75.0009766, 29.6175537, -104.7179489, 104.8624725
41: -54.2000809, 25.7927094, -54.2052422, 25.7934723, -79.9935455, 79.9979553
42: -38.8467026, 29.3171730, -38.7221451, 29.2700157, -68.1167068, 68.0393219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=206, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 695

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.3481789, upper bound: 38.9133793
time: 82.48 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.3481789, upper bound: 38.9133842
time: 75.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -53.2390213, 42.8929558, -53.0500107, 42.4992752, -95.7382965, 95.9429626
1: -31.5715675, 35.9603424, -31.4255905, 35.6495132, -67.2210770, 67.3859329
2: -30.3491936, 35.5246391, -30.2023106, 35.0835419, -65.4327240, 65.7269363
3: -33.8948898, 41.5134163, -33.8028679, 41.0918465, -74.9867401, 75.3162842
4: -40.0039673, 38.8173561, -39.8421440, 38.3375740, -78.3415375, 78.6595001
5: -36.8434982, 41.2896843, -36.7413330, 40.7829132, -77.6264038, 78.0310211
6: -55.9153519, 22.4170227, -55.8414536, 22.2341099, -78.1494598, 78.2584763
7: -42.8214569, 40.0230217, -42.5770569, 39.3773727, -82.1988297, 82.6000824
8: -39.2559280, 45.3587227, -39.1304016, 44.8572998, -84.1132278, 84.4891205
9: -34.1674995, 37.4599800, -33.9196777, 37.2177429, -71.3852386, 71.3796539
10: -55.1442795, 52.3625374, -54.6868210, 52.0225983, -107.1668701, 107.0493622
11: -56.4512825, 39.6744690, -56.3547859, 39.6179848, -96.0692673, 96.0292511
12: -58.9945221, 43.9653358, -57.8748703, 43.5701904, -102.5647125, 101.8402100
13: -48.7112274, 49.5660820, -48.2212601, 49.2216072, -97.9328156, 97.7873383
14: -81.2816162, 43.3446198, -80.4263763, 43.0555191, -124.3371277, 123.7709885
15: -40.2587013, 36.3488159, -39.9061356, 36.2547150, -76.5134125, 76.2549438
16: -58.2529182, 40.7073555, -58.0050926, 40.3420258, -98.5949402, 98.7124405
17: -85.0527344, 62.5110931, -84.0965652, 62.1335487, -147.1862793, 146.6076660
18: -48.9403687, 29.1030865, -48.6284790, 28.9955006, -77.9358673, 77.7315674
19: -41.3507576, 19.4409695, -41.1882858, 19.4306068, -60.7813568, 60.6292534
20: -35.3729324, 21.8024521, -35.1077423, 21.7023487, -57.0752792, 56.9101944
21: -49.1571960, 25.4424400, -48.9158325, 25.3762951, -74.5334930, 74.3582764
22: -50.8201828, 29.9838543, -50.1006584, 29.7566109, -80.5767975, 80.0845108
23: -39.1208420, 26.5838604, -38.9152679, 26.5609169, -65.6817627, 65.4991302
24: -45.1775360, 22.8165283, -45.0112801, 22.7083549, -67.8858871, 67.8278046
25: -38.4388390, 30.9518414, -38.1980362, 30.8274117, -69.2662506, 69.1498795
26: -58.9047546, 37.5366745, -57.8605270, 37.2371979, -96.1419449, 95.3972015
27: -49.3679657, 27.3551273, -49.0989723, 27.1426010, -76.5105667, 76.4541016
28: -37.8339157, 28.8102264, -37.6518402, 28.7522202, -66.5861359, 66.4620667
29: -55.2936287, 34.3055801, -54.6017342, 34.0601730, -89.3537979, 88.9073181
30: -47.7327194, 27.2136230, -47.6150703, 27.1165943, -74.8493042, 74.8286896
31: -48.9936523, 23.9800034, -48.7780037, 23.7475185, -72.7411652, 72.7580109
32: -49.1345482, 27.4537582, -48.8528023, 27.2941399, -76.4286880, 76.3065643
33: -71.8454132, 43.9127998, -71.6465607, 43.6299057, -115.4753113, 115.5593567
34: -60.8876610, 29.9670868, -60.7400513, 29.9163837, -90.8040466, 90.7071381
35: -57.1596298, 34.6320953, -57.0422897, 34.5991058, -91.7587280, 91.6743851
36: -57.2174416, 33.8742981, -56.8175697, 33.7319183, -90.9493561, 90.6918640
37: -85.1737747, 32.9710312, -84.9685135, 32.9473610, -118.1211395, 117.9395447
38: -69.0786285, 40.9719238, -68.7828674, 40.8280754, -109.9067078, 109.7547836
39: -85.0315933, 40.6747665, -84.8385849, 40.4222336, -125.4538269, 125.5133514
40: -75.1858521, 29.9115105, -74.9269180, 29.4735813, -104.6594238, 104.8384247
41: -54.2917709, 25.8810215, -54.1595421, 25.7520828, -80.0438538, 80.0405579
42: -38.9105759, 29.3921280, -38.6623611, 29.2328300, -68.1434021, 68.0544891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=206, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=405, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 695

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5190852, upper bound: 38.9875010
time: 81.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5835177, upper bound: 38.9875730
time: 81.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -53.2390213, 42.8929558, -53.1465340, 42.6686630, -95.9076843, 96.0394821
1: -31.5715675, 35.9603424, -31.5008354, 35.7941933, -67.3657532, 67.4611816
2: -30.3491936, 35.5246391, -30.2816639, 35.2581673, -65.6073608, 65.8062897
3: -33.8948898, 41.5134163, -33.8600311, 41.2520142, -75.1468964, 75.3734436
4: -40.0039673, 38.8173561, -39.9227600, 38.5204391, -78.5244064, 78.7401123
5: -36.8434982, 41.2896843, -36.8100510, 40.9803276, -77.8238220, 78.0997314
6: -55.9153519, 22.4170227, -55.8632126, 22.3123150, -78.2276688, 78.2802353
7: -42.8214569, 40.0230217, -42.7020035, 39.6500435, -82.4714966, 82.7250214
8: -39.2559280, 45.3587227, -39.2162437, 45.0694656, -84.3253937, 84.5749588
9: -34.1674995, 37.4599800, -34.0198936, 37.2979050, -71.4654083, 71.4798737
10: -55.1442795, 52.3625374, -54.8677826, 52.1435776, -107.2878494, 107.2303162
11: -56.4512825, 39.6744690, -56.3854446, 39.6467857, -96.0980530, 96.0599136
12: -58.9945221, 43.9653358, -58.2473984, 43.7109413, -102.7054596, 102.2127380
13: -48.7112274, 49.5660820, -48.4005280, 49.3371429, -98.0483704, 97.9666138
14: -81.2816162, 43.3446198, -80.7428894, 43.1745491, -124.4561615, 124.0875015
15: -40.2587013, 36.3488159, -40.0403748, 36.2908859, -76.5495911, 76.3891907
16: -58.2529182, 40.7073555, -58.0745697, 40.4510574, -98.7039642, 98.7819214
17: -85.0527344, 62.5110931, -84.3903503, 62.2535934, -147.3063202, 146.9014435
18: -48.9403687, 29.1030865, -48.7424507, 29.0279293, -77.9682999, 77.8455353
19: -41.3507576, 19.4409695, -41.2548027, 19.4468765, -60.7976341, 60.6957703
20: -35.3729324, 21.8024521, -35.1896515, 21.7367535, -57.1096878, 56.9921036
21: -49.1571960, 25.4424400, -49.0078163, 25.4151611, -74.5723572, 74.4502563
22: -50.8201828, 29.9838543, -50.3602829, 29.8580170, -80.6781998, 80.3441391
23: -39.1208420, 26.5838604, -38.9956894, 26.5899429, -65.7107849, 65.5795517
24: -45.1775360, 22.8165283, -45.0772972, 22.7505665, -67.9281006, 67.8938217
25: -38.4388390, 30.9518414, -38.3102036, 30.9037266, -69.3425674, 69.2620468
26: -58.9047546, 37.5366745, -58.2230835, 37.3653526, -96.2700958, 95.7597580
27: -49.3679657, 27.3551273, -49.2008400, 27.2299442, -76.5979080, 76.5559692
28: -37.8339157, 28.8102264, -37.7287407, 28.7924919, -66.6264038, 66.5389709
29: -55.2936287, 34.3055801, -54.8649521, 34.1575737, -89.4512024, 89.1705322
30: -47.7327194, 27.2136230, -47.6673279, 27.1607418, -74.8934631, 74.8809509
31: -48.9936523, 23.9800034, -48.8581505, 23.8182716, -72.8119202, 72.8381500
32: -49.1345482, 27.4537582, -48.9453049, 27.3395271, -76.4740753, 76.3990631
33: -71.8454132, 43.9127998, -71.7205353, 43.7189941, -115.5643921, 115.6333313
34: -60.8876610, 29.9670868, -60.8061295, 29.9446564, -90.8323212, 90.7732086
35: -57.1596298, 34.6320953, -57.0901413, 34.6129723, -91.7725983, 91.7222366
36: -57.2174416, 33.8742981, -56.9518776, 33.7929840, -91.0104218, 90.8261719
37: -85.1737747, 32.9710312, -85.0702438, 32.9908981, -118.1646729, 118.0412750
38: -69.0786285, 40.9719238, -68.8703537, 40.8631439, -109.9417725, 109.8422775
39: -85.0315933, 40.6747665, -84.9019470, 40.4774208, -125.5090179, 125.5767059
40: -75.1858521, 29.9115105, -75.0176010, 29.6329842, -104.8188324, 104.9291077
41: -54.2917709, 25.8810215, -54.2176590, 25.8226871, -80.1144562, 80.0986786
42: -38.9105759, 29.3921280, -38.7333832, 29.2935772, -68.2041550, 68.1255112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=206, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 695

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5190852, upper bound: 38.9903336
time: 124.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5835177, upper bound: 38.9903976
time: 66.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -53.1268005, 42.8964729, -53.1372032, 42.6946945, -95.8214951, 96.0336761
1: -31.4656506, 35.9398079, -31.4759979, 35.7913857, -67.2570343, 67.4158020
2: -30.2322178, 35.5057983, -30.2425842, 35.2472687, -65.4794846, 65.7483826
3: -33.7420235, 41.4632874, -33.8054504, 41.2585335, -75.0005569, 75.2687378
4: -39.8735809, 38.7974930, -39.8850517, 38.5492859, -78.4228668, 78.6825409
5: -36.6937790, 41.2438202, -36.7386093, 40.9541130, -77.6478882, 77.9824219
6: -55.8763695, 22.3918304, -55.8768692, 22.2828598, -78.1592255, 78.2686996
7: -42.6859589, 40.0262566, -42.6397057, 39.6122398, -82.2982025, 82.6659622
8: -39.1327744, 45.3596344, -39.1842880, 45.0954590, -84.2282257, 84.5439224
9: -34.0469055, 37.4411697, -33.9230804, 37.3094635, -71.3563614, 71.3642502
10: -55.0861130, 52.2722931, -54.7719803, 52.1041527, -107.1902618, 107.0442734
11: -56.3853416, 39.5726509, -56.4228745, 39.6162758, -96.0016174, 95.9955292
12: -59.0372009, 43.8577309, -58.2681885, 43.6960144, -102.7332153, 102.1259155
13: -48.6312370, 49.5030632, -48.3401031, 49.3215141, -97.9527512, 97.8431702
14: -81.2935028, 43.2503662, -80.7555389, 43.1465225, -124.4400177, 124.0059052
15: -40.2082558, 36.3163223, -39.9694138, 36.2402420, -76.4485016, 76.2857361
16: -58.1393204, 40.7506561, -58.0647774, 40.5350609, -98.6743774, 98.8154297
17: -85.0903549, 62.4104691, -84.5070190, 62.2618637, -147.3522034, 146.9174805
18: -48.9112511, 28.9687271, -48.7552338, 28.9952660, -77.9065170, 77.7239609
19: -41.2846985, 19.3344975, -41.2485466, 19.4012775, -60.6859741, 60.5830460
20: -35.3596420, 21.7148151, -35.2251205, 21.7213326, -57.0809708, 56.9399338
21: -49.1053162, 25.3418159, -49.0147552, 25.3752651, -74.4805756, 74.3565674
22: -50.8495407, 29.8793125, -50.4411507, 29.8271065, -80.6766434, 80.3204651
23: -39.0418396, 26.4424973, -39.0162506, 26.5364838, -65.5783234, 65.4587479
24: -45.1014404, 22.7020760, -45.0985947, 22.6951141, -67.7965546, 67.8006668
25: -38.4054260, 30.8310318, -38.3411293, 30.8634911, -69.2689133, 69.1721573
26: -58.9478455, 37.3660622, -58.2710571, 37.3096886, -96.2575226, 95.6371155
27: -49.3064270, 27.2707958, -49.1809120, 27.1608086, -76.4672318, 76.4517059
28: -37.7728348, 28.6867962, -37.7348251, 28.7499962, -66.5228271, 66.4216156
29: -55.3093452, 34.2021637, -54.9677467, 34.1385002, -89.4478455, 89.1699066
30: -47.6847076, 27.1369019, -47.6965866, 27.1504040, -74.8350983, 74.8334808
31: -48.8963814, 23.8754120, -48.8373032, 23.7501278, -72.6465073, 72.7127151
32: -49.0890160, 27.4013195, -48.9424210, 27.3304977, -76.4195099, 76.3437424
33: -71.7973022, 43.8915405, -71.7355957, 43.7707367, -115.5680313, 115.6271362
34: -60.8200607, 29.8614197, -60.8015442, 29.9475098, -90.7675705, 90.6629639
35: -57.1215057, 34.5705338, -57.1059265, 34.6270218, -91.7485275, 91.6764603
36: -57.2216721, 33.7889786, -56.9834213, 33.7837219, -91.0053864, 90.7723999
37: -85.0640488, 32.8460541, -85.0752487, 32.9498215, -118.0138702, 117.9212952
38: -69.0372162, 40.8588562, -68.8884125, 40.8511810, -109.8883820, 109.7472687
39: -84.9499130, 40.6641617, -84.9193573, 40.5668983, -125.5167999, 125.5835190
40: -75.1289215, 29.9378910, -75.0201721, 29.6568127, -104.7857208, 104.9580612
41: -54.2233276, 25.8224545, -54.1995125, 25.7936211, -80.0169373, 80.0219650
42: -38.8818130, 29.3340874, -38.7385406, 29.2799149, -68.1617279, 68.0726318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=206, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 695

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5051827, upper bound: 38.9778976
time: 66.25 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5695912, upper bound: 38.9779290
time: 73.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -53.1268005, 42.8964729, -53.2357826, 42.8655167, -95.9923172, 96.1322556
1: -31.4656506, 35.9398079, -31.5525703, 35.9382858, -67.4039307, 67.4923782
2: -30.2322178, 35.5057983, -30.3224468, 35.4229774, -65.6551971, 65.8282471
3: -33.7420235, 41.4632874, -33.8634148, 41.4214630, -75.1634827, 75.3267059
4: -39.8735809, 38.7974930, -39.9663315, 38.7339401, -78.6075134, 78.7638245
5: -36.6937790, 41.2438202, -36.8077888, 41.1535797, -77.8473511, 78.0516052
6: -55.8763695, 22.3918304, -55.9053917, 22.3634262, -78.2397919, 78.2972183
7: -42.6859589, 40.0262566, -42.7667732, 39.8856087, -82.5715561, 82.7930298
8: -39.1327744, 45.3596344, -39.2711945, 45.3094063, -84.4421768, 84.6308289
9: -34.0469055, 37.4411697, -34.0247383, 37.3912888, -71.4381943, 71.4659119
10: -55.0861130, 52.2722931, -54.9555359, 52.2254486, -107.3115463, 107.2278290
11: -56.3853416, 39.5726509, -56.4634094, 39.6455727, -96.0309143, 96.0360565
12: -59.0372009, 43.8577309, -58.6432037, 43.8383331, -102.8755188, 102.5009308
13: -48.6312370, 49.5030632, -48.5280418, 49.4527016, -98.0839386, 98.0310974
14: -81.2935028, 43.2503662, -81.0748901, 43.2660294, -124.5595322, 124.3252563
15: -40.2082558, 36.3163223, -40.1191750, 36.2979660, -76.5062180, 76.4355011
16: -58.1393204, 40.7506561, -58.1458511, 40.6535110, -98.7928314, 98.8965073
17: -85.0903549, 62.4104691, -84.8029175, 62.3830185, -147.4733734, 147.2133789
18: -48.9112511, 28.9687271, -48.8704147, 29.0289421, -77.9401932, 77.8391342
19: -41.2846985, 19.3344975, -41.3172112, 19.4190941, -60.7037926, 60.6517067
20: -35.3596420, 21.7148151, -35.3096237, 21.7564545, -57.1160965, 57.0244370
21: -49.1053162, 25.3418159, -49.1108170, 25.4144726, -74.5197830, 74.4526291
22: -50.8495407, 29.8793125, -50.7087097, 29.9302082, -80.7797394, 80.5880203
23: -39.0418396, 26.4424973, -39.1012993, 26.5661812, -65.6080170, 65.5437927
24: -45.1014404, 22.7020760, -45.1676178, 22.7383728, -67.8398132, 67.8696899
25: -38.4054260, 30.8310318, -38.4642334, 30.9458046, -69.3512115, 69.2952652
26: -58.9478455, 37.3660622, -58.6369629, 37.4393005, -96.3871307, 96.0030212
27: -49.3064270, 27.2707958, -49.2878189, 27.2501373, -76.5565643, 76.5586090
28: -37.7728348, 28.6867962, -37.8136902, 28.7913895, -66.5642166, 66.5004883
29: -55.3093452, 34.2021637, -55.2322159, 34.2368813, -89.5462189, 89.4343796
30: -47.6847076, 27.1369019, -47.7541275, 27.1951847, -74.8798904, 74.8910294
31: -48.8963814, 23.8754120, -48.9215622, 23.8237305, -72.7201080, 72.7969742
32: -49.0890160, 27.4013195, -49.0375443, 27.3779774, -76.4669952, 76.4388580
33: -71.7973022, 43.8915405, -71.8114319, 43.8634644, -115.6607513, 115.7029724
34: -60.8200607, 29.8614197, -60.8692551, 29.9804039, -90.8004608, 90.7306671
35: -57.1215057, 34.5705338, -57.1556358, 34.6476784, -91.7691803, 91.7261658
36: -57.2216721, 33.7889786, -57.1230087, 33.8465576, -91.0682220, 90.9119873
37: -85.0640488, 32.8460541, -85.1799622, 32.9957352, -118.0597839, 118.0260086
38: -69.0372162, 40.8588562, -68.9781570, 40.8888054, -109.9260101, 109.8370132
39: -84.9499130, 40.6641617, -84.9852142, 40.6285744, -125.5784912, 125.6493759
40: -75.1289215, 29.9378910, -75.1171570, 29.8203163, -104.9492264, 105.0550461
41: -54.2233276, 25.8224545, -54.2627792, 25.8681908, -80.0915146, 80.0852356
42: -38.8818130, 29.3340874, -38.8155212, 29.3419418, -68.2237549, 68.1496124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=206, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 695

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5051827, upper bound: 38.9800167
time: 96.17 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5695912, upper bound: 38.9800496
time: 102.32 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 201.03 seconds
IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 201.03
Output dim: 2, lower bound: -38.4460657, upper bound: 38.9772307
IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 201.03
Output dim: 2, lower bound: -38.3481789, upper bound: 38.9064174
IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 201.03
Output dim: 2, lower bound: -38.3481789, upper bound: 38.9133793
IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 11, time: 201.03
Output dim: 2, lower bound: -38.3481789, upper bound: 38.9133842
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 201.03
Output dim: 2, lower bound: -38.5190852, upper bound: 38.9875010
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 201.03
Output dim: 2, lower bound: -38.5835177, upper bound: 38.9875730
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 201.03
Output dim: 2, lower bound: -38.5190852, upper bound: 38.9903336
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 201.03
Output dim: 2, lower bound: -38.5835177, upper bound: 38.9903976
IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 201.03
Output dim: 2, lower bound: -38.5051827, upper bound: 38.9778976
IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 201.03
Output dim: 2, lower bound: -38.5695912, upper bound: 38.9779290
IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 201.03
Output dim: 2, lower bound: -38.5051827, upper bound: 38.9800167
IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 201.03
Output dim: 2, lower bound: -38.5695912, upper bound: 38.9800496
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 201.03
Output dim: 2, lower bound: -38.6449095, upper bound: 38.9903443
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 201.03
Output dim: 2, lower bound: -38.6449095, upper bound: 38.9925126
IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 201.03
Output dim: 2, lower bound: -38.5312020, upper bound: 38.9896817
IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 201.03
Output dim: 2, lower bound: -38.5312020, upper bound: 38.9911905
IS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 201.03
Output dim: 2, lower bound: -38.5882087, upper bound: 38.9896822
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 201.03
Output dim: 2, lower bound: -38.6619142, upper bound: 39.0001417
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 201.03
Output dim: 2, lower bound: -38.6619145, upper bound: 39.0016522

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 103.79 + 7250.15 = 7353.94 seconds
