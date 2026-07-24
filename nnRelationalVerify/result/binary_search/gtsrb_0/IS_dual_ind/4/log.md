## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 113.951405901
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094)
1: (-49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728)
2: (-48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477)
3: (-52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180)
4: (-65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904)
5: (-55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736)
6: (-86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349)
7: (-67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533)
8: (-78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369)
9: (-61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378)
10: (-91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857)
11: (-83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308)
12: (-60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689)
13: (-67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264)
14: (-116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854)
15: (-66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329)
16: (-97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562)
17: (-109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844)
18: (-91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036)
19: (-67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806)
20: (-65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906)
21: (-84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383)
22: (-74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077)
23: (-66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527)
24: (-85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511)
25: (-63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815)
26: (-90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034)
27: (-99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819)
28: (-66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736)
29: (-78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234)
30: (-83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396)
31: (-89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892)
32: (-76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247)
33: (-113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937)
34: (-90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572)
35: (-88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323)
36: (-87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383)
37: (-136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212)
38: (-109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264)
39: (-121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669)
40: (-112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069)
41: (-86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731)
42: (-58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549)

## BASE Result
execution time: IAR + LP analysis = 2.98 + 156.46 = 159.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -123.2820485, upper bound: 123.2820485


# Binary Search by BASE starts (time budget: 17840.56 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=172.8270263671875
rel_dist={13: [-114.05422291258427, 114.05422290962252]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=172.8270263671875
rel_dist={13: [-107.09440620076711, 107.0944062017914]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=172.8270263671875
rel_dist={13: [-109.64944808549234, 109.64944808379073]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=172.8270263671875
rel_dist={13: [-111.95995320337546, 111.95995319383253]}

## Binary Search Result
Binary search time: 507.25 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual_ind) starts
Time budget: 17333.30 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1725

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1070378, upper bound: 119.2543410
time: 280.01 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1070378, upper bound: 119.2543410
time: 105.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 385.32 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 385.32
Output dim: 13, lower bound: -119.1070378, upper bound: 119.2543410
IS_A2, status: Status.UNKNOWN, split count: 1, time: 385.32
Output dim: 13, lower bound: -119.1070378, upper bound: 119.2543410

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -103.2789230, 67.2919617, -103.4275055, 67.3309326, -170.6098633, 170.7194672
1: -49.8466568, 50.5750504, -49.9396400, 50.6054420, -100.4520874, 100.5146790
2: -48.4513588, 49.9596710, -48.5514336, 49.9882736, -98.4396286, 98.5110931
3: -52.0105362, 63.1161232, -52.1370926, 63.1526947, -115.1632309, 115.2532196
4: -65.4317474, 56.8847351, -65.5805969, 56.9162445, -122.3479919, 122.4653320
5: -55.1246986, 58.8373146, -55.2269096, 58.8788719, -114.0035706, 114.0642242
6: -86.2812881, 50.4314461, -86.3505859, 50.4982758, -136.7795563, 136.7820129
7: -67.1984406, 49.0380058, -67.3165283, 49.0751266, -116.2735596, 116.3545380
8: -78.5159607, 69.6070328, -78.6722717, 69.6544876, -148.1704407, 148.2792969
9: -61.5627823, 67.9424744, -61.6534195, 67.9786987, -129.5414734, 129.5958862
10: -91.5939331, 89.5898285, -91.6681137, 89.6522903, -181.2462158, 181.2579346
11: -83.5207901, 42.2501907, -83.5645218, 42.3625565, -125.8833466, 125.8147125
12: -60.2727776, 76.4904327, -60.3532066, 76.5679474, -136.8407288, 136.8436279
13: -67.5933228, 104.8646698, -67.8876038, 104.9086838, -172.5020142, 172.7522736
14: -116.6980896, 60.6436386, -116.8340988, 60.6754417, -177.3735352, 177.4777374
15: -66.7402344, 63.3593292, -66.8231354, 63.4267578, -130.1669922, 130.1824646
16: -97.4204788, 53.8608360, -97.4929352, 53.9063454, -151.3268280, 151.3537598
17: -109.5036087, 72.3375397, -109.6876831, 72.3759918, -181.8795776, 182.0252228
18: -90.9898911, 46.4713249, -91.0392456, 46.6717224, -137.6615906, 137.5105591
19: -67.0177841, 34.8906326, -67.0548553, 35.0132523, -102.0310364, 101.9454880
20: -65.6934586, 42.5447807, -65.7299805, 42.6568756, -108.3503342, 108.2747650
21: -84.9193420, 42.7168999, -84.9625626, 42.8643837, -127.7837067, 127.6794586
22: -74.9042816, 62.0042725, -74.9573212, 62.1264687, -137.0307465, 136.9615936
23: -66.8766403, 48.0928650, -66.9087906, 48.2453613, -115.1220016, 115.0016556
24: -85.6265869, 55.3728523, -85.6700974, 55.5336189, -141.1601868, 141.0429382
25: -63.0124893, 57.7699051, -63.0536919, 57.9233665, -120.9358521, 120.8235931
26: -90.5337219, 57.2201385, -90.5988007, 57.4068222, -147.9405518, 147.8189392
27: -99.0048828, 45.8935585, -99.0494385, 46.0894127, -145.0942993, 144.9429932
28: -66.5087280, 53.1532402, -66.5430374, 53.3034630, -119.8121948, 119.6962738
29: -78.0079727, 65.3635406, -78.0570297, 65.4883575, -143.4963226, 143.4205627
30: -83.1591110, 55.8056870, -83.1986847, 55.9409866, -139.1000977, 139.0043640
31: -89.7233124, 48.8365593, -89.7729721, 49.0060349, -138.7293396, 138.6095276
32: -76.6843872, 59.4214249, -76.7857819, 59.4707336, -136.1551056, 136.2072144
33: -113.1405029, 70.8516159, -113.2786865, 70.8990173, -184.0395203, 184.1303101
34: -90.0883026, 52.1685448, -90.1741867, 52.1998672, -142.2881622, 142.3427277
35: -88.5957031, 63.7896461, -88.6917267, 63.8301468, -152.4258423, 152.4813690
36: -87.6762085, 60.5937653, -87.7749023, 60.6254082, -148.3016052, 148.3686676
37: -136.0542297, 45.2893219, -136.1687622, 45.3311234, -181.3853455, 181.4580841
38: -109.3837280, 68.5697327, -109.4971924, 68.6113434, -177.9950714, 178.0669250
39: -121.7112503, 68.2278595, -121.9318314, 68.2526703, -189.9638977, 190.1596680
40: -112.6575012, 35.4271545, -112.7601318, 35.4607010, -148.1181946, 148.1872864
41: -85.9746704, 49.4464340, -86.0473251, 49.4933929, -135.4680634, 135.4937592
42: -58.2059669, 46.8197632, -58.2585373, 46.8810234, -105.0869904, 105.0783005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=461, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 576

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1725

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1070378, upper bound: 119.1070378
time: 105.93 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1070378, upper bound: 119.2543410
time: 91.90 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -103.4987564, 67.4807739, -103.4361420, 67.3330841, -170.8318176, 170.9169159
1: -49.9841461, 50.7235031, -49.9451141, 50.6069641, -100.5911102, 100.6686096
2: -48.5848351, 50.1187897, -48.5588188, 49.9897461, -98.5745697, 98.6776123
3: -52.1724319, 63.3241463, -52.1455307, 63.1546364, -115.3270721, 115.4696808
4: -65.6181717, 57.0589294, -65.5884018, 56.9177132, -122.5358810, 122.6473236
5: -55.2560272, 59.0489693, -55.2327194, 58.8810959, -114.1371002, 114.2816925
6: -86.4045105, 50.5677490, -86.3539658, 50.5025406, -136.9070435, 136.9217224
7: -67.3689041, 49.2245445, -67.3233948, 49.0771942, -116.4460983, 116.5479355
8: -78.7240067, 69.8168335, -78.6821289, 69.6571808, -148.3811951, 148.4989624
9: -61.7092667, 68.0905609, -61.6578369, 67.9802399, -129.6895142, 129.7483826
10: -91.7617569, 89.7050629, -91.6726074, 89.6552048, -181.4169312, 181.3776703
11: -83.7840271, 42.4016991, -83.5662613, 42.3699913, -126.1540222, 125.9679565
12: -60.4028091, 76.6851578, -60.3574982, 76.5723724, -136.9751892, 137.0426483
13: -67.9509506, 105.2626648, -67.9076691, 104.9107513, -172.8616943, 173.1703339
14: -116.9430313, 60.7728233, -116.8412781, 60.6769524, -177.6199799, 177.6141052
15: -66.8750763, 63.4576836, -66.8281708, 63.4292221, -130.3042908, 130.2858582
16: -97.5842438, 53.9723434, -97.4968872, 53.9085350, -151.4927673, 151.4692383
17: -109.7997818, 72.6318970, -109.6989365, 72.3781052, -182.1778717, 182.3308258
18: -91.2873077, 46.7206841, -91.0417480, 46.6851387, -137.9724426, 137.7624207
19: -67.2232437, 35.0353203, -67.0563812, 35.0215569, -102.2447968, 102.0917053
20: -65.8555374, 42.6894531, -65.7313919, 42.6640091, -108.5195312, 108.4208450
21: -85.1621933, 42.8966904, -84.9639206, 42.8741837, -128.0363770, 127.8606110
22: -75.1213989, 62.1596489, -74.9597931, 62.1344757, -137.2558746, 137.1194458
23: -67.1238098, 48.2861977, -66.9096832, 48.2556534, -115.3794632, 115.1958771
24: -85.8645935, 55.5689545, -85.6717072, 55.5445023, -141.4090881, 141.2406616
25: -63.2279701, 57.9655228, -63.0553894, 57.9335594, -121.1615295, 121.0209122
26: -90.8253021, 57.4471397, -90.6019058, 57.4191132, -148.2444153, 148.0490417
27: -99.2961731, 46.1256485, -99.0510559, 46.1027527, -145.3989258, 145.1766968
28: -66.7207642, 53.3386803, -66.5443268, 53.3135567, -120.0343170, 119.8829956
29: -78.2435379, 65.5223541, -78.0590897, 65.4969482, -143.7404785, 143.5814514
30: -83.3387451, 55.9877357, -83.2001038, 55.9498100, -139.2885437, 139.1878357
31: -90.0051270, 49.0414886, -89.7751083, 49.0173874, -139.0224915, 138.8165894
32: -76.8434982, 59.5563354, -76.7912445, 59.4737396, -136.3172302, 136.3475800
33: -113.3452225, 71.0796661, -113.2865448, 70.9015808, -184.2467804, 184.3662109
34: -90.2239685, 52.2682648, -90.1779022, 52.2011719, -142.4251404, 142.4461670
35: -88.7385101, 63.9412918, -88.6956024, 63.8325424, -152.5710449, 152.6368713
36: -87.8224945, 60.6886368, -87.7792664, 60.6271172, -148.4496155, 148.4678955
37: -136.2642822, 45.4085770, -136.1746826, 45.3334846, -181.5977631, 181.5832520
38: -109.5997543, 68.6971359, -109.5039444, 68.6136551, -178.2134094, 178.2010803
39: -122.0251160, 68.4777374, -121.9469070, 68.2538147, -190.2789307, 190.4246368
40: -112.8514175, 35.5672913, -112.7657166, 35.4621811, -148.3135986, 148.3330078
41: -86.1057053, 49.5208855, -86.0512924, 49.4955864, -135.6012878, 135.5721741
42: -58.3013573, 46.9149704, -58.2613564, 46.8838997, -105.1852417, 105.1763306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=461, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1592

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0991097, upper bound: 119.0859723
time: 127.13 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0998849, upper bound: 119.2471849
time: 232.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 361.96 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 361.96
Output dim: 13, lower bound: -119.1070378, upper bound: 119.1070378
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 361.96
Output dim: 13, lower bound: -119.1070378, upper bound: 119.2543410
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 361.96
Output dim: 13, lower bound: -119.0991097, upper bound: 119.0859723
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 361.96
Output dim: 13, lower bound: -119.0998849, upper bound: 119.2471849

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -103.2789230, 67.2919617, -103.2789230, 67.2919617, -170.5708771, 170.5708618
1: -49.8466568, 50.5750504, -49.8466568, 50.5750504, -100.4216995, 100.4217072
2: -48.4513588, 49.9596710, -48.4513588, 49.9596710, -98.4110260, 98.4110260
3: -52.0105362, 63.1161232, -52.0105362, 63.1161232, -115.1266632, 115.1266632
4: -65.4317474, 56.8847351, -65.4317474, 56.8847351, -122.3164825, 122.3164825
5: -55.1246986, 58.8373146, -55.1246986, 58.8373146, -113.9620132, 113.9620132
6: -86.2812881, 50.4314461, -86.2812881, 50.4314461, -136.7127075, 136.7127228
7: -67.1984406, 49.0380058, -67.1984406, 49.0380058, -116.2364502, 116.2364502
8: -78.5159607, 69.6070328, -78.5159607, 69.6070328, -148.1229858, 148.1229858
9: -61.5627823, 67.9424744, -61.5627823, 67.9424744, -129.5052490, 129.5052490
10: -91.5939331, 89.5898285, -91.5939331, 89.5898285, -181.1837311, 181.1837463
11: -83.5207901, 42.2501907, -83.5207901, 42.2501907, -125.7709808, 125.7709808
12: -60.2727776, 76.4904327, -60.2727776, 76.4904327, -136.7631989, 136.7632141
13: -67.5933228, 104.8646698, -67.5933228, 104.8646698, -172.4579926, 172.4579926
14: -116.6980896, 60.6436386, -116.6980896, 60.6436386, -177.3417358, 177.3417358
15: -66.7402344, 63.3593292, -66.7402344, 63.3593292, -130.0995636, 130.0995636
16: -97.4204788, 53.8608360, -97.4204788, 53.8608360, -151.2813110, 151.2813110
17: -109.5036087, 72.3375397, -109.5036087, 72.3375397, -181.8411560, 181.8411407
18: -90.9898911, 46.4713249, -90.9898911, 46.4713249, -137.4612122, 137.4611969
19: -67.0177841, 34.8906326, -67.0177841, 34.8906326, -101.9084167, 101.9084167
20: -65.6934586, 42.5447807, -65.6934586, 42.5447807, -108.2382278, 108.2382355
21: -84.9193420, 42.7168999, -84.9193420, 42.7168999, -127.6362457, 127.6362457
22: -74.9042816, 62.0042725, -74.9042816, 62.0042725, -136.9085388, 136.9085541
23: -66.8766403, 48.0928650, -66.8766403, 48.0928650, -114.9695053, 114.9695053
24: -85.6265869, 55.3728523, -85.6265869, 55.3728523, -140.9994202, 140.9994354
25: -63.0124893, 57.7699051, -63.0124893, 57.7699051, -120.7823944, 120.7823944
26: -90.5337219, 57.2201385, -90.5337219, 57.2201385, -147.7538605, 147.7538605
27: -99.0048828, 45.8935585, -99.0048828, 45.8935585, -144.8984375, 144.8984375
28: -66.5087280, 53.1532402, -66.5087280, 53.1532402, -119.6619720, 119.6619720
29: -78.0079727, 65.3635406, -78.0079727, 65.3635406, -143.3715210, 143.3715210
30: -83.1591110, 55.8056870, -83.1591110, 55.8056870, -138.9647827, 138.9647827
31: -89.7233124, 48.8365593, -89.7233124, 48.8365593, -138.5598755, 138.5598755
32: -76.6843872, 59.4214249, -76.6843872, 59.4214249, -136.1058044, 136.1058044
33: -113.1405029, 70.8516159, -113.1405029, 70.8516159, -183.9921112, 183.9921265
34: -90.0883026, 52.1685448, -90.0883026, 52.1685448, -142.2568512, 142.2568359
35: -88.5957031, 63.7896461, -88.5957031, 63.7896461, -152.3853455, 152.3853455
36: -87.6762085, 60.5937653, -87.6762085, 60.5937653, -148.2699585, 148.2699738
37: -136.0542297, 45.2893219, -136.0542297, 45.2893219, -181.3435516, 181.3435516
38: -109.3837280, 68.5697327, -109.3837280, 68.5697327, -177.9534607, 177.9534607
39: -121.7112503, 68.2278595, -121.7112503, 68.2278595, -189.9391174, 189.9391174
40: -112.6575012, 35.4271545, -112.6575012, 35.4271545, -148.0846558, 148.0846558
41: -85.9746704, 49.4464340, -85.9746704, 49.4464340, -135.4211121, 135.4210968
42: -58.2059669, 46.8197632, -58.2059669, 46.8197632, -105.0257263, 105.0257263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 576

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1592

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.9385123, upper bound: 119.1993772
time: 119.42 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0998846, upper bound: 119.2001692
time: 134.09 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -103.2789230, 67.2919617, -103.4987564, 67.4807739, -170.7597046, 170.7907104
1: -49.8466568, 50.5750504, -49.9841461, 50.7235031, -100.5701599, 100.5591888
2: -48.4513588, 49.9596710, -48.5848351, 50.1187897, -98.5701294, 98.5445099
3: -52.0105362, 63.1161232, -52.1724319, 63.3241463, -115.3346786, 115.2885590
4: -65.4317474, 56.8847351, -65.6181717, 57.0589294, -122.4906616, 122.5029068
5: -55.1246986, 58.8373146, -55.2560272, 59.0489693, -114.1736679, 114.0933380
6: -86.2812881, 50.4314461, -86.4045105, 50.5677490, -136.8490295, 136.8359528
7: -67.1984406, 49.0380058, -67.3689041, 49.2245445, -116.4229736, 116.4069061
8: -78.5159607, 69.6070328, -78.7240067, 69.8168335, -148.3327789, 148.3310242
9: -61.5627823, 67.9424744, -61.7092667, 68.0905609, -129.6533356, 129.6517334
10: -91.5939331, 89.5898285, -91.7617569, 89.7050629, -181.2989960, 181.3515930
11: -83.5207901, 42.2501907, -83.7840271, 42.4016991, -125.9224854, 126.0342178
12: -60.2727776, 76.4904327, -60.4028091, 76.6851578, -136.9579315, 136.8932495
13: -67.5933228, 104.8646698, -67.9509506, 105.2626648, -172.8559875, 172.8156128
14: -116.6980896, 60.6436386, -116.9430313, 60.7728233, -177.4709167, 177.5866699
15: -66.7402344, 63.3593292, -66.8750763, 63.4576836, -130.1979218, 130.2344055
16: -97.4204788, 53.8608360, -97.5842438, 53.9723434, -151.3928223, 151.4450836
17: -109.5036087, 72.3375397, -109.7997818, 72.6318970, -182.1354828, 182.1373138
18: -90.9898911, 46.4713249, -91.2873077, 46.7206841, -137.7105408, 137.7586212
19: -67.0177841, 34.8906326, -67.2232437, 35.0353203, -102.0530930, 102.1138763
20: -65.6934586, 42.5447807, -65.8555374, 42.6894531, -108.3829117, 108.4003067
21: -84.9193420, 42.7168999, -85.1621933, 42.8966904, -127.8160324, 127.8790894
22: -74.9042816, 62.0042725, -75.1213989, 62.1596489, -137.0639343, 137.1256714
23: -66.8766403, 48.0928650, -67.1238098, 48.2861977, -115.1628342, 115.2166748
24: -85.6265869, 55.3728523, -85.8645935, 55.5689545, -141.1955414, 141.2374420
25: -63.0124893, 57.7699051, -63.2279701, 57.9655228, -120.9780121, 120.9978790
26: -90.5337219, 57.2201385, -90.8253021, 57.4471397, -147.9808655, 148.0454407
27: -99.0048828, 45.8935585, -99.2961731, 46.1256485, -145.1305237, 145.1897278
28: -66.5087280, 53.1532402, -66.7207642, 53.3386803, -119.8473892, 119.8740082
29: -78.0079727, 65.3635406, -78.2435379, 65.5223541, -143.5303345, 143.6070557
30: -83.1591110, 55.8056870, -83.3387451, 55.9877357, -139.1468506, 139.1444244
31: -89.7233124, 48.8365593, -90.0051270, 49.0414886, -138.7648010, 138.8416748
32: -76.6843872, 59.4214249, -76.8434982, 59.5563354, -136.2407227, 136.2649231
33: -113.1405029, 70.8516159, -113.3452225, 71.0796661, -184.2201538, 184.1968384
34: -90.0883026, 52.1685448, -90.2239685, 52.2682648, -142.3565674, 142.3925018
35: -88.5957031, 63.7896461, -88.7385101, 63.9412918, -152.5369873, 152.5281525
36: -87.6762085, 60.5937653, -87.8224945, 60.6886368, -148.3648376, 148.4162598
37: -136.0542297, 45.2893219, -136.2642822, 45.4085770, -181.4627686, 181.5536041
38: -109.3837280, 68.5697327, -109.5997543, 68.6971359, -178.0808563, 178.1694946
39: -121.7112503, 68.2278595, -122.0251160, 68.4777374, -190.1889954, 190.2529602
40: -112.6575012, 35.4271545, -112.8514175, 35.5672913, -148.2247925, 148.2785645
41: -85.9746704, 49.4464340, -86.1057053, 49.5208855, -135.4955444, 135.5521393
42: -58.2059669, 46.8197632, -58.3013573, 46.9149704, -105.1209412, 105.1211243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=460, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 576

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1592

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.9385123, upper bound: 119.2463807
time: 332.70 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0998846, upper bound: 119.2471852
time: 98.60 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -103.4949112, 67.4788818, -103.3691254, 67.2995758, -170.7944946, 170.8480072
1: -49.9804077, 50.7224083, -49.8792229, 50.5874329, -100.5678253, 100.6016312
2: -48.5814934, 50.1174545, -48.5004654, 49.9662247, -98.5477142, 98.6179199
3: -52.1674347, 63.3229866, -52.0585861, 63.1341667, -115.3015823, 115.3815689
4: -65.6157990, 57.0574684, -65.5466003, 56.8919144, -122.5077133, 122.6040649
5: -55.2519264, 59.0477448, -55.1621094, 58.8591652, -114.1110916, 114.2098465
6: -86.3999557, 50.5660019, -86.2728119, 50.4724464, -136.8724060, 136.8388062
7: -67.3630219, 49.2234726, -67.2211914, 49.0574493, -116.4204712, 116.4446640
8: -78.7218933, 69.8146057, -78.6453934, 69.6175232, -148.3394012, 148.4599915
9: -61.7047310, 68.0894699, -61.5804100, 67.9612732, -129.6660004, 129.6698761
10: -91.7541351, 89.7029343, -91.5377808, 89.6178131, -181.3719330, 181.2407074
11: -83.7801437, 42.4000397, -83.4982758, 42.3405533, -126.1206818, 125.8983154
12: -60.3984756, 76.6824951, -60.2801208, 76.5258484, -136.9243164, 136.9626160
13: -67.9385071, 105.2607269, -67.6866531, 104.8769684, -172.8154755, 172.9473877
14: -116.9395905, 60.7674751, -116.7811432, 60.5823212, -177.5219116, 177.5486145
15: -66.8726654, 63.4522629, -66.7858124, 63.3336601, -130.2063293, 130.2380676
16: -97.5779343, 53.9703827, -97.3865356, 53.8736115, -151.4515228, 151.3569183
17: -109.7958450, 72.6281891, -109.6301270, 72.3124847, -182.1083221, 182.2583008
18: -91.2853088, 46.7075844, -91.0069885, 46.4602127, -137.7455139, 137.7145538
19: -67.2213058, 35.0316467, -67.0222092, 34.9565926, -102.1779022, 102.0538559
20: -65.8537445, 42.6841354, -65.7002258, 42.5690918, -108.4228287, 108.3843613
21: -85.1595993, 42.8918419, -84.9178772, 42.7881012, -127.9476929, 127.8097229
22: -75.1190796, 62.1504364, -74.9187241, 61.9700546, -137.0891418, 137.0691528
23: -67.1222992, 48.2816353, -66.8835297, 48.1747017, -115.2969971, 115.1651611
24: -85.8628922, 55.5616493, -85.6421204, 55.4153900, -141.2782898, 141.2037659
25: -63.2263412, 57.9597206, -63.0270882, 57.8331795, -121.0595245, 120.9868011
26: -90.8231125, 57.4346962, -90.5635605, 57.1978683, -148.0209808, 147.9982605
27: -99.2942429, 46.1150513, -99.0170746, 45.9140434, -145.2082825, 145.1321259
28: -66.7191925, 53.3315620, -66.5172882, 53.1866074, -119.9057922, 119.8488464
29: -78.2414780, 65.5149918, -78.0221405, 65.3654785, -143.6069489, 143.5371399
30: -83.3364868, 55.9817200, -83.1602631, 55.8452377, -139.1817322, 139.1419678
31: -90.0026779, 49.0371513, -89.7319717, 48.9416542, -138.9443207, 138.7691193
32: -76.8375854, 59.5548248, -76.6889343, 59.4468269, -136.2844086, 136.2437592
33: -113.3395309, 71.0773773, -113.1865311, 70.8615494, -184.2010651, 184.2639160
34: -90.2203217, 52.2663383, -90.1139450, 52.1679382, -142.3882599, 142.3802795
35: -88.7334442, 63.9397736, -88.6082001, 63.8052673, -152.5387115, 152.5479736
36: -87.8184280, 60.6875877, -87.7080002, 60.6090012, -148.4274292, 148.3955841
37: -136.2616577, 45.4065094, -136.1284180, 45.2966232, -181.5582886, 181.5349274
38: -109.5950851, 68.6952286, -109.4212570, 68.5808716, -178.1759644, 178.1164856
39: -122.0156937, 68.4757996, -121.7798767, 68.2200775, -190.2357788, 190.2556763
40: -112.8487473, 35.5649376, -112.7188339, 35.4211235, -148.2698669, 148.2837677
41: -86.1002274, 49.5192947, -85.9546509, 49.4668121, -135.5670319, 135.4739380
42: -58.2965164, 46.9131012, -58.1758080, 46.8513641, -105.1478729, 105.0888977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 576

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 693

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2420110, upper bound: 118.9686016
time: 114.51 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2420110, upper bound: 119.0815840
time: 146.38 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -103.4945831, 67.4795990, -103.4878311, 67.3711548, -170.8657379, 170.9674377
1: -49.9820862, 50.7227402, -49.9606171, 50.6751404, -100.6572266, 100.6833496
2: -48.5827980, 50.1180344, -48.5749245, 50.0497055, -98.6324997, 98.6929626
3: -52.1693573, 63.3230171, -52.1776505, 63.2420731, -115.4114151, 115.5006714
4: -65.6168747, 57.0579529, -65.6163406, 56.9673195, -122.5841980, 122.6742935
5: -55.2532425, 59.0480728, -55.2538567, 58.9633179, -114.2165604, 114.3019257
6: -86.4023590, 50.5665855, -86.3947906, 50.6303062, -137.0326233, 136.9613647
7: -67.3655548, 49.2238426, -67.3410950, 49.1638794, -116.5294342, 116.5649414
8: -78.7228851, 69.8152924, -78.7231979, 69.7020035, -148.4248962, 148.5384827
9: -61.7073059, 68.0896759, -61.6902122, 68.0756226, -129.7829285, 129.7798767
10: -91.7583313, 89.7037582, -91.7093658, 89.8270874, -181.5854187, 181.4131165
11: -83.7815781, 42.4005203, -83.6239243, 42.4084549, -126.1900330, 126.0244446
12: -60.4007149, 76.6839142, -60.3889656, 76.7005768, -137.1012878, 137.0728760
13: -67.9459076, 105.2612152, -67.9386749, 105.1664124, -173.1123199, 173.1998901
14: -116.9409256, 60.7689590, -116.9346390, 60.6956902, -177.6366119, 177.7035980
15: -66.8736115, 63.4552765, -66.9241638, 63.4571686, -130.3307648, 130.3794403
16: -97.5784225, 53.9711227, -97.5376205, 54.0648460, -151.6432495, 151.5087433
17: -109.7968979, 72.6282959, -109.7774429, 72.4201813, -182.2170715, 182.4057312
18: -91.2856598, 46.7151947, -91.2442474, 46.7117310, -137.9973907, 137.9594421
19: -67.2220001, 35.0337219, -67.1425476, 35.0374298, -102.2594299, 102.1762695
20: -65.8544846, 42.6871643, -65.8327789, 42.6764717, -108.5309601, 108.5199432
21: -85.1605301, 42.8944817, -85.0650406, 42.8842239, -128.0447540, 127.9595184
22: -75.1198120, 62.1560745, -75.1812592, 62.1473694, -137.2671814, 137.3373413
23: -67.1226349, 48.2841606, -67.0292206, 48.2701187, -115.3927460, 115.3133850
24: -85.8630371, 55.5660019, -85.8496094, 55.5604744, -141.4234924, 141.4156036
25: -63.2269135, 57.9628487, -63.1795158, 57.9467392, -121.1736526, 121.1423569
26: -90.8237305, 57.4421959, -90.8567352, 57.4402122, -148.2639313, 148.2989197
27: -99.2948380, 46.1211319, -99.2855225, 46.1150322, -145.4098511, 145.4066467
28: -66.7198639, 53.3356552, -66.7015839, 53.3237801, -120.0436401, 120.0372391
29: -78.2418137, 65.5192413, -78.2557297, 65.5062103, -143.7480164, 143.7749481
30: -83.3373032, 55.9848709, -83.3330536, 55.9635696, -139.3008575, 139.3179169
31: -90.0034409, 49.0380211, -89.8807602, 49.0374641, -139.0408936, 138.9187775
32: -76.8412399, 59.5553436, -76.8438721, 59.5598068, -136.4010468, 136.3992004
33: -113.3421478, 71.0782776, -113.3254471, 71.0035248, -184.3456726, 184.4037170
34: -90.2216034, 52.2670364, -90.2178955, 52.2481689, -142.4697571, 142.4849243
35: -88.7361145, 63.9400749, -88.7305069, 63.9171906, -152.6533051, 152.6705780
36: -87.8206253, 60.6879921, -87.8193665, 60.6670685, -148.4877014, 148.5073547
37: -136.2623901, 45.4074593, -136.2411499, 45.3855820, -181.6479797, 181.6486053
38: -109.5972443, 68.6958771, -109.5427628, 68.7163086, -178.3135376, 178.2386475
39: -122.0212402, 68.4765396, -121.9966736, 68.4226074, -190.4438171, 190.4732056
40: -112.8500824, 35.5656281, -112.8490753, 35.5217514, -148.3718262, 148.4147034
41: -86.1031342, 49.5199776, -86.0955811, 49.5733795, -135.6765137, 135.6155548
42: -58.2991447, 46.9137344, -58.3061409, 46.9940109, -105.2931366, 105.2198639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 693

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0954496, upper bound: 119.1299959
time: 105.50 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2428069, upper bound: 119.2428065
time: 105.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 213.86 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 213.86
Output dim: 13, lower bound: -118.9385123, upper bound: 119.1993772
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 213.86
Output dim: 13, lower bound: -119.0998846, upper bound: 119.2001692
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 213.86
Output dim: 13, lower bound: -118.9385123, upper bound: 119.2463807
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 213.86
Output dim: 13, lower bound: -119.0998846, upper bound: 119.2471852
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 213.86
Output dim: 13, lower bound: -119.2420110, upper bound: 118.9686016
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 213.86
Output dim: 13, lower bound: -119.2420110, upper bound: 119.0815840
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 213.86
Output dim: 13, lower bound: -119.0954496, upper bound: 119.1299959
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 213.86
Output dim: 13, lower bound: -119.2428069, upper bound: 119.2428065

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -103.2118759, 67.2584229, -103.2750931, 67.2900543, -170.5019226, 170.5335083
1: -49.7807770, 50.5555000, -49.8429070, 50.5739441, -100.3547211, 100.3984070
2: -48.3930740, 49.9361420, -48.4480247, 49.9583511, -98.3514252, 98.3841629
3: -51.9236069, 63.0956917, -52.0055351, 63.1149559, -115.0385513, 115.1012268
4: -65.3899689, 56.8589325, -65.4293671, 56.8832588, -122.2732239, 122.2882996
5: -55.0540695, 58.8153343, -55.1206360, 58.8360519, -113.8901215, 113.9359665
6: -86.2001343, 50.4013252, -86.2767181, 50.4297371, -136.6298676, 136.6780396
7: -67.0962219, 49.0182304, -67.1925430, 49.0368919, -116.1331177, 116.2107697
8: -78.4792786, 69.5673523, -78.5139008, 69.6048050, -148.0840759, 148.0812378
9: -61.4853668, 67.9235077, -61.5582581, 67.9413757, -129.4267426, 129.4817657
10: -91.4590988, 89.5523682, -91.5863495, 89.5876617, -181.0467529, 181.1387177
11: -83.4527206, 42.2207108, -83.5168762, 42.2485390, -125.7012482, 125.7375870
12: -60.1954384, 76.4438324, -60.2684135, 76.4876938, -136.6831360, 136.7122345
13: -67.3724365, 104.8308411, -67.5809631, 104.8627243, -172.2351532, 172.4118042
14: -116.6380844, 60.5489731, -116.6946640, 60.6382904, -177.2763672, 177.2436371
15: -66.6978073, 63.2638588, -66.7378082, 63.3538895, -130.0516968, 130.0016632
16: -97.3101044, 53.8259468, -97.4141998, 53.8588638, -151.1689758, 151.2401428
17: -109.4348450, 72.2719498, -109.4996567, 72.3337860, -181.7686310, 181.7716064
18: -90.9551392, 46.2464523, -90.9879074, 46.4581757, -137.4132996, 137.2343597
19: -66.9835739, 34.8256721, -67.0158386, 34.8869972, -101.8705597, 101.8415070
20: -65.6622696, 42.4498291, -65.6916733, 42.5394402, -108.2017059, 108.1415024
21: -84.8732834, 42.6308441, -84.9166794, 42.7120628, -127.5853424, 127.5475235
22: -74.8631744, 61.8398895, -74.9019318, 61.9950409, -136.8582153, 136.7418213
23: -66.8504868, 48.0119171, -66.8751373, 48.0882874, -114.9387741, 114.8870468
24: -85.5969696, 55.2437248, -85.6249161, 55.3655052, -140.9624634, 140.8686371
25: -62.9841614, 57.6695786, -63.0108757, 57.7641068, -120.7482681, 120.6804504
26: -90.4953918, 56.9989433, -90.5315552, 57.2077370, -147.7031250, 147.5305023
27: -98.9708633, 45.7048569, -99.0028992, 45.8830070, -144.8538513, 144.7077637
28: -66.4816818, 53.0263405, -66.5071640, 53.1461220, -119.6278000, 119.5335007
29: -77.9710236, 65.2321320, -78.0058746, 65.3561707, -143.3271942, 143.2380066
30: -83.1193237, 55.7011185, -83.1568146, 55.7996826, -138.9190063, 138.8579254
31: -89.6801453, 48.7608643, -89.7208786, 48.8322678, -138.5124054, 138.4817505
32: -76.5820999, 59.3944969, -76.6784668, 59.4198723, -136.0019684, 136.0729523
33: -113.0406342, 70.8115082, -113.1348343, 70.8493423, -183.8899841, 183.9463348
34: -90.0243378, 52.1352539, -90.0846939, 52.1666222, -142.1909637, 142.2199402
35: -88.5084229, 63.7622986, -88.5906372, 63.7880669, -152.2964783, 152.3529358
36: -87.6049652, 60.5756493, -87.6721268, 60.5927582, -148.1977234, 148.2477722
37: -136.0079193, 45.2524147, -136.0515747, 45.2871857, -181.2951050, 181.3039856
38: -109.3009949, 68.5369034, -109.3790359, 68.5678482, -177.8688354, 177.9159393
39: -121.5443497, 68.1940918, -121.7018509, 68.2258987, -189.7702332, 189.8959351
40: -112.6105804, 35.3860779, -112.6548080, 35.4248009, -148.0353851, 148.0408936
41: -85.8780518, 49.4176788, -85.9692230, 49.4448242, -135.3228760, 135.3868866
42: -58.1204033, 46.7872353, -58.2011147, 46.8178940, -104.9382935, 104.9883423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 693

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.9225054, upper bound: 119.1949155
time: 108.28 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0347361, upper bound: 119.1949155
time: 131.28 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -103.3306274, 67.3300018, -103.2747574, 67.2907715, -170.6213989, 170.6047668
1: -49.8622055, 50.6432266, -49.8445892, 50.5742912, -100.4364929, 100.4878159
2: -48.4675331, 50.0196228, -48.4493065, 49.9589157, -98.4264450, 98.4689331
3: -52.0426750, 63.2035599, -52.0074501, 63.1150246, -115.1576996, 115.2110138
4: -65.4597092, 56.9343224, -65.4304657, 56.8837814, -122.3434753, 122.3647842
5: -55.1458435, 58.9195862, -55.1219254, 58.8364372, -113.9822845, 114.0415039
6: -86.3220901, 50.5592613, -86.2791138, 50.4303131, -136.7524109, 136.8383789
7: -67.2161026, 49.1246910, -67.1950684, 49.0372620, -116.2533646, 116.3197632
8: -78.5571213, 69.6518555, -78.5148621, 69.6054688, -148.1625824, 148.1667175
9: -61.5950851, 68.0378876, -61.5608253, 67.9415741, -129.5366516, 129.5987091
10: -91.6306152, 89.7617035, -91.5905151, 89.5885162, -181.2191162, 181.3522186
11: -83.5783768, 42.2886543, -83.5183105, 42.2489853, -125.8273621, 125.8069611
12: -60.3042793, 76.6186142, -60.2706413, 76.4891434, -136.7934113, 136.8892517
13: -67.6244202, 105.1203079, -67.5883484, 104.8632355, -172.4876251, 172.7086487
14: -116.7915802, 60.6623840, -116.6959610, 60.6398087, -177.4313965, 177.3583374
15: -66.8363037, 63.3873367, -66.7387695, 63.3569603, -130.1932678, 130.1260986
16: -97.4610748, 54.0171776, -97.4146957, 53.8596382, -151.3207092, 151.4318695
17: -109.5822372, 72.3795929, -109.5007172, 72.3339081, -181.9161377, 181.8803101
18: -91.1924362, 46.4979401, -90.9882660, 46.4658203, -137.6582642, 137.4862061
19: -67.1039581, 34.9065247, -67.0165253, 34.8890762, -101.9930344, 101.9230499
20: -65.7948608, 42.5572205, -65.6923981, 42.5424652, -108.3373260, 108.2496185
21: -85.0204315, 42.7269402, -84.9176559, 42.7146988, -127.7351303, 127.6445847
22: -75.1257782, 62.0172081, -74.9027100, 62.0007286, -137.1265106, 136.9199219
23: -66.9962006, 48.1073074, -66.8754807, 48.0908318, -115.0870361, 114.9827881
24: -85.8045197, 55.3887825, -85.6250153, 55.3698654, -141.1743774, 141.0137939
25: -63.1366081, 57.7831230, -63.0114746, 57.7672462, -120.9038544, 120.7945938
26: -90.7885895, 57.2412376, -90.5321884, 57.2151947, -148.0037842, 147.7734222
27: -99.2393799, 45.9058075, -99.0035172, 45.8890610, -145.1284485, 144.9093323
28: -66.6659851, 53.1634941, -66.5078430, 53.1502533, -119.8162308, 119.6713333
29: -78.2046051, 65.3728027, -78.0062103, 65.3603821, -143.5649872, 143.3790131
30: -83.2921143, 55.8194656, -83.1576843, 55.8027840, -139.0948944, 138.9771423
31: -89.8289490, 48.8566780, -89.7216568, 48.8331032, -138.6620483, 138.5783386
32: -76.7369690, 59.5075378, -76.6821136, 59.4203949, -136.1573486, 136.1896362
33: -113.1794586, 70.9535217, -113.1374207, 70.8502274, -184.0296936, 184.0909424
34: -90.1282349, 52.2154732, -90.0859680, 52.1673088, -142.2955322, 142.3014374
35: -88.6305923, 63.8742943, -88.5933456, 63.7883949, -152.4189758, 152.4676361
36: -87.7162628, 60.6337395, -87.6743011, 60.5931206, -148.3093567, 148.3080444
37: -136.1206360, 45.3414154, -136.0523071, 45.2881699, -181.4087830, 181.3937225
38: -109.4224854, 68.6723633, -109.3812180, 68.5684967, -177.9909821, 178.0535889
39: -121.7610931, 68.3965912, -121.7073898, 68.2266998, -189.9877930, 190.1039734
40: -112.7409668, 35.4867439, -112.6561737, 35.4254608, -148.1664276, 148.1429138
41: -86.0189209, 49.5242653, -85.9721146, 49.4455261, -135.4644318, 135.4963684
42: -58.2506981, 46.9299316, -58.2037430, 46.8185196, -105.0692139, 105.1336746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 576

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 693

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0838677, upper bound: 119.1957043
time: 100.94 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1957038, upper bound: 119.1957043
time: 100.44 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -103.2118759, 67.2584229, -103.4949112, 67.4788818, -170.6907654, 170.7533264
1: -49.7807770, 50.5555000, -49.9804077, 50.7224083, -100.5031891, 100.5358963
2: -48.3930740, 49.9361420, -48.5814934, 50.1174545, -98.5105286, 98.5176392
3: -51.9236069, 63.0956917, -52.1674347, 63.3229866, -115.2465820, 115.2631226
4: -65.3899689, 56.8589325, -65.6157990, 57.0574684, -122.4474335, 122.4747314
5: -55.0540695, 58.8153343, -55.2519264, 59.0477448, -114.1018143, 114.0672607
6: -86.2001343, 50.4013252, -86.3999557, 50.5660019, -136.7661438, 136.8012695
7: -67.0962219, 49.0182304, -67.3630219, 49.2234726, -116.3196945, 116.3812408
8: -78.4792786, 69.5673523, -78.7218933, 69.8146057, -148.2938690, 148.2892303
9: -61.4853668, 67.9235077, -61.7047310, 68.0894699, -129.5748291, 129.6282349
10: -91.4590988, 89.5523682, -91.7541351, 89.7029343, -181.1620178, 181.3065033
11: -83.4527206, 42.2207108, -83.7801437, 42.4000397, -125.8527451, 126.0008392
12: -60.1954384, 76.4438324, -60.3984756, 76.6824951, -136.8779297, 136.8423157
13: -67.3724365, 104.8308411, -67.9385071, 105.2607269, -172.6331635, 172.7693481
14: -116.6380844, 60.5489731, -116.9395905, 60.7674751, -177.4055634, 177.4885559
15: -66.6978073, 63.2638588, -66.8726654, 63.4522629, -130.1500702, 130.1365204
16: -97.3101044, 53.8259468, -97.5779343, 53.9703827, -151.2804871, 151.4038696
17: -109.4348450, 72.2719498, -109.7958450, 72.6281891, -182.0630341, 182.0677948
18: -90.9551392, 46.2464523, -91.2853088, 46.7075844, -137.6626892, 137.5317535
19: -66.9835739, 34.8256721, -67.2213058, 35.0316467, -102.0152206, 102.0469666
20: -65.6622696, 42.4498291, -65.8537445, 42.6841354, -108.3464050, 108.3035660
21: -84.8732834, 42.6308441, -85.1595993, 42.8918419, -127.7651215, 127.7904358
22: -74.8631744, 61.8398895, -75.1190796, 62.1504364, -137.0136108, 136.9589691
23: -66.8504868, 48.0119171, -67.1222992, 48.2816353, -115.1321259, 115.1342163
24: -85.5969696, 55.2437248, -85.8628922, 55.5616493, -141.1585999, 141.1066132
25: -62.9841614, 57.6695786, -63.2263412, 57.9597206, -120.9438782, 120.8959122
26: -90.4953918, 56.9989433, -90.8231125, 57.4346962, -147.9300842, 147.8220520
27: -98.9708633, 45.7048569, -99.2942429, 46.1150513, -145.0859070, 144.9990997
28: -66.4816818, 53.0263405, -66.7191925, 53.3315620, -119.8132477, 119.7455292
29: -77.9710236, 65.2321320, -78.2414780, 65.5149918, -143.4860229, 143.4736023
30: -83.1193237, 55.7011185, -83.3364868, 55.9817200, -139.1010437, 139.0375977
31: -89.6801453, 48.7608643, -90.0026779, 49.0371513, -138.7172852, 138.7635498
32: -76.5820999, 59.3944969, -76.8375854, 59.5548248, -136.1369171, 136.2320862
33: -113.0406342, 70.8115082, -113.3395309, 71.0773773, -184.1180115, 184.1510315
34: -90.0243378, 52.1352539, -90.2203217, 52.2663383, -142.2906799, 142.3555603
35: -88.5084229, 63.7622986, -88.7334442, 63.9397736, -152.4481812, 152.4957428
36: -87.6049652, 60.5756493, -87.8184280, 60.6875877, -148.2925568, 148.3940735
37: -136.0079193, 45.2524147, -136.2616577, 45.4065094, -181.4144287, 181.5140686
38: -109.3009949, 68.5369034, -109.5950851, 68.6952286, -177.9962158, 178.1319885
39: -121.5443497, 68.1940918, -122.0156937, 68.4757996, -190.0201416, 190.2097778
40: -112.6105804, 35.3860779, -112.8487473, 35.5649376, -148.1755066, 148.2348328
41: -85.8780518, 49.4176788, -86.1002274, 49.5192947, -135.3973389, 135.5179138
42: -58.1204033, 46.7872353, -58.2965164, 46.9131012, -105.0335083, 105.0837402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 693

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.8216133, upper bound: 119.2420110
time: 91.57 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.9340658, upper bound: 119.2420110
time: 111.63 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -103.3306274, 67.3300018, -103.4945831, 67.4795990, -170.8102264, 170.8245850
1: -49.8622055, 50.6432266, -49.9820862, 50.7227402, -100.5849457, 100.6253128
2: -48.4675331, 50.0196228, -48.5827980, 50.1180344, -98.5855484, 98.6024170
3: -52.0426750, 63.2035599, -52.1693573, 63.3230171, -115.3656921, 115.3729172
4: -65.4597092, 56.9343224, -65.6168747, 57.0579529, -122.5176620, 122.5511932
5: -55.1458435, 58.9195862, -55.2532425, 59.0480728, -114.1939163, 114.1728134
6: -86.3220901, 50.5592613, -86.4023590, 50.5665855, -136.8886719, 136.9616089
7: -67.2161026, 49.1246910, -67.3655548, 49.2238426, -116.4399414, 116.4902496
8: -78.5571213, 69.6518555, -78.7228851, 69.8152924, -148.3724060, 148.3747406
9: -61.5950851, 68.0378876, -61.7073059, 68.0896759, -129.6847534, 129.7451782
10: -91.6306152, 89.7617035, -91.7583313, 89.7037582, -181.3343658, 181.5200348
11: -83.5783768, 42.2886543, -83.7815781, 42.4005203, -125.9788971, 126.0702362
12: -60.3042793, 76.6186142, -60.4007149, 76.6839142, -136.9881744, 137.0193329
13: -67.6244202, 105.1203079, -67.9459076, 105.2612152, -172.8856201, 173.0662079
14: -116.7915802, 60.6623840, -116.9409256, 60.7689590, -177.5605469, 177.6033020
15: -66.8363037, 63.3873367, -66.8736115, 63.4552765, -130.2915802, 130.2609558
16: -97.4610748, 54.0171776, -97.5784225, 53.9711227, -151.4321899, 151.5955963
17: -109.5822372, 72.3795929, -109.7968979, 72.6282959, -182.2105408, 182.1764832
18: -91.1924362, 46.4979401, -91.2856598, 46.7151947, -137.9076233, 137.7835999
19: -67.1039581, 34.9065247, -67.2220001, 35.0337219, -102.1376648, 102.1285248
20: -65.7948608, 42.5572205, -65.8544846, 42.6871643, -108.4820251, 108.4117050
21: -85.0204315, 42.7269402, -85.1605301, 42.8944817, -127.9149170, 127.8874664
22: -75.1257782, 62.0172081, -75.1198120, 62.1560745, -137.2818604, 137.1370239
23: -66.9962006, 48.1073074, -67.1226349, 48.2841606, -115.2803650, 115.2299423
24: -85.8045197, 55.3887825, -85.8630371, 55.5660019, -141.3704987, 141.2518158
25: -63.1366081, 57.7831230, -63.2269135, 57.9628487, -121.0994568, 121.0100327
26: -90.7885895, 57.2412376, -90.8237305, 57.4421959, -148.2307892, 148.0649567
27: -99.2393799, 45.9058075, -99.2948380, 46.1211319, -145.3605042, 145.2006226
28: -66.6659851, 53.1634941, -66.7198639, 53.3356552, -120.0016403, 119.8833618
29: -78.2046051, 65.3728027, -78.2418137, 65.5192413, -143.7238312, 143.6146240
30: -83.2921143, 55.8194656, -83.3373032, 55.9848709, -139.2769775, 139.1567688
31: -89.8289490, 48.8566780, -90.0034409, 49.0380211, -138.8669739, 138.8601227
32: -76.7369690, 59.5075378, -76.8412399, 59.5553436, -136.2922974, 136.3487854
33: -113.1794586, 70.9535217, -113.3421478, 71.0782776, -184.2577209, 184.2956543
34: -90.1282349, 52.2154732, -90.2216034, 52.2670364, -142.3952637, 142.4370728
35: -88.6305923, 63.8742943, -88.7361145, 63.9400749, -152.5706635, 152.6104126
36: -87.7162628, 60.6337395, -87.8206253, 60.6879921, -148.4042358, 148.4543610
37: -136.1206360, 45.3414154, -136.2623901, 45.4074593, -181.5280914, 181.6038055
38: -109.4224854, 68.6723633, -109.5972443, 68.6958771, -178.1183624, 178.2695923
39: -121.7610931, 68.3965912, -122.0212402, 68.4765396, -190.2376404, 190.4178162
40: -112.7409668, 35.4867439, -112.8500824, 35.5656281, -148.3065948, 148.3368225
41: -86.0189209, 49.5242653, -86.1031342, 49.5199776, -135.5388947, 135.6273956
42: -58.2506981, 46.9299316, -58.2991447, 46.9137344, -105.1644287, 105.2290726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 693

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.9831332, upper bound: 119.2428069
time: 110.25 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0954490, upper bound: 119.2428068
time: 162.31 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -103.3445053, 67.4183884, -103.3459778, 67.2904892, -170.6349792, 170.7643738
1: -49.8705368, 50.6510162, -49.8629265, 50.5768547, -100.4473877, 100.5139465
2: -48.4999352, 50.0653992, -48.4876137, 49.9583244, -98.4582520, 98.5530090
3: -52.0502853, 63.2398682, -52.0408020, 63.1215782, -115.1718597, 115.2806702
4: -65.5433426, 56.9825287, -65.5349960, 56.8805923, -122.4239349, 122.5175095
5: -55.0791779, 58.9809723, -55.1364288, 58.8490448, -113.9282227, 114.1174011
6: -86.3331680, 50.4849510, -86.2626495, 50.4596786, -136.7928467, 136.7475891
7: -67.1452179, 49.1692162, -67.1889496, 49.0491562, -116.1943741, 116.3581696
8: -78.6031799, 69.7405090, -78.6265259, 69.6065521, -148.2097321, 148.3670349
9: -61.5651245, 68.0213470, -61.5580215, 67.9508438, -129.5159607, 129.5793762
10: -91.4650116, 89.6180801, -91.4932175, 89.6048584, -181.0698547, 181.1112823
11: -83.4170380, 42.3297882, -83.4446182, 42.3292923, -125.7463303, 125.7744064
12: -60.3106766, 76.4140778, -60.2668877, 76.4861603, -136.7968292, 136.6809387
13: -67.8173904, 104.9745331, -67.6679993, 104.8352814, -172.6526794, 172.6425171
14: -116.7623978, 60.7173653, -116.7537994, 60.5743332, -177.3367310, 177.4711609
15: -66.7731171, 63.3955269, -66.7704773, 63.3250008, -130.0980988, 130.1660004
16: -97.3144302, 53.8952179, -97.3466187, 53.8623352, -151.1767578, 151.2418365
17: -109.5185852, 72.3544998, -109.5881653, 72.2691879, -181.7877808, 181.9426575
18: -91.0918121, 46.6088638, -90.9784622, 46.4438095, -137.5356140, 137.5873108
19: -67.0537415, 34.9827003, -66.9963684, 34.9492340, -102.0029755, 101.9790573
20: -65.6548462, 42.6205597, -65.6715012, 42.5591507, -108.2139969, 108.2920609
21: -84.8660049, 42.8290634, -84.8734818, 42.7784042, -127.6444092, 127.7025299
22: -74.9280624, 62.0857544, -74.8901367, 61.9603043, -136.8883667, 136.9758911
23: -66.9767303, 48.2216644, -66.8621368, 48.1654282, -115.1421585, 115.0837936
24: -85.6472931, 55.4850464, -85.6107941, 55.4038353, -141.0511169, 141.0958405
25: -63.0539322, 57.8976707, -63.0014000, 57.8238068, -120.8777237, 120.8990707
26: -90.6847992, 57.3453064, -90.5430298, 57.1836281, -147.8684235, 147.8883209
27: -99.0299683, 46.0321426, -98.9787903, 45.9011688, -144.9311371, 145.0109253
28: -66.6287231, 53.2668495, -66.5039978, 53.1765442, -119.8052597, 119.7708435
29: -77.9380188, 65.4589767, -77.9775238, 65.3569946, -143.2950134, 143.4364929
30: -83.0499191, 55.9044151, -83.1187668, 55.8328171, -138.8827362, 139.0231781
31: -89.7815475, 48.9690666, -89.6976471, 48.9314499, -138.7129974, 138.6667175
32: -76.7590256, 59.4332085, -76.6770172, 59.4284286, -136.1874390, 136.1102295
33: -113.2583694, 70.7915573, -113.1737823, 70.8192596, -184.0776367, 183.9653320
34: -90.1576843, 51.9280586, -90.1041870, 52.1130180, -142.2706909, 142.0322266
35: -88.6614532, 63.5776939, -88.5970459, 63.7482452, -152.4096985, 152.1747131
36: -87.7523346, 60.4032974, -87.6978302, 60.5639572, -148.3162842, 148.1011353
37: -136.1341705, 45.1387024, -136.1089172, 45.2578049, -181.3919678, 181.2476196
38: -109.5184784, 68.2678833, -109.4092789, 68.5124817, -178.0309448, 177.6771545
39: -121.9094772, 68.1620331, -121.7629852, 68.1743622, -190.0838318, 189.9250183
40: -112.7312393, 35.4223938, -112.7009125, 35.4000244, -148.1312561, 148.1233063
41: -86.0412292, 49.3394928, -85.9456482, 49.4402924, -135.4815063, 135.2851410
42: -58.2099953, 46.8440704, -58.1627007, 46.8407440, -105.0507355, 105.0067596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=604, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0872453, upper bound: 118.7970483
time: 88.54 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0872453, upper bound: 118.9624394
time: 198.09 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -103.5427170, 67.4991455, -103.3645706, 67.2960358, -170.8387451, 170.8637085
1: -49.9925194, 50.7521400, -49.8770676, 50.5843544, -100.5768738, 100.6292114
2: -48.5933304, 50.1270828, -48.4974442, 49.9634399, -98.5567703, 98.6245270
3: -52.1707344, 63.3309937, -52.0553055, 63.1253433, -115.2960815, 115.3862991
4: -65.6400986, 57.0743980, -65.5443726, 56.8875084, -122.5276031, 122.6187592
5: -55.2552338, 59.0915489, -55.1593246, 58.8559341, -114.1111603, 114.2508621
6: -86.4454803, 50.5863495, -86.2683716, 50.4699898, -136.9154510, 136.8547058
7: -67.3772202, 49.2909050, -67.2183685, 49.0554504, -116.4326630, 116.5092773
8: -78.7311096, 69.8787689, -78.6419525, 69.6143494, -148.3454437, 148.5207214
9: -61.7178650, 68.1214600, -61.5754128, 67.9592133, -129.6770782, 129.6968689
10: -91.7503815, 89.7386856, -91.5286636, 89.6144867, -181.3648529, 181.2673492
11: -83.7926865, 42.4875526, -83.4920044, 42.3377380, -126.1304092, 125.9795532
12: -60.5915642, 76.6871033, -60.2772675, 76.5222168, -137.1137848, 136.9643707
13: -68.1476593, 105.2616577, -67.6826630, 104.8723602, -173.0200195, 172.9443207
14: -116.9923401, 60.7806702, -116.7767944, 60.5794563, -177.5717773, 177.5574646
15: -66.8839264, 63.4858971, -66.7830505, 63.3300591, -130.2139893, 130.2689514
16: -97.5977707, 54.0246048, -97.3786087, 53.8716354, -151.4693909, 151.4032135
17: -109.8357239, 72.6313629, -109.6252289, 72.3054657, -182.1411896, 182.2565765
18: -91.2965393, 46.7449226, -91.0019531, 46.4561081, -137.7526245, 137.7468719
19: -67.2282715, 35.0572891, -67.0173492, 34.9545135, -102.1827850, 102.0746384
20: -65.8729095, 42.8210373, -65.6974640, 42.5663452, -108.4392471, 108.5184937
21: -85.1847916, 43.0483932, -84.9129486, 42.7850571, -127.9698410, 127.9613419
22: -75.1782303, 62.2203102, -74.9151154, 61.9671516, -137.1453857, 137.1354065
23: -67.1342926, 48.3450546, -66.8801727, 48.1725616, -115.3068542, 115.2252274
24: -85.8843231, 55.6818924, -85.6371689, 55.4115791, -141.2958832, 141.3190613
25: -63.2539177, 58.0524979, -63.0239906, 57.8305702, -121.0844879, 121.0764923
26: -90.8682098, 57.4994011, -90.5599823, 57.1946831, -148.0628815, 148.0593872
27: -99.3045197, 46.2892075, -99.0114746, 45.9107170, -145.2152100, 145.3006897
28: -66.7238464, 53.3667374, -66.5147247, 53.1848488, -119.9086914, 119.8814621
29: -78.2911911, 65.6283264, -78.0172272, 65.3632812, -143.6544800, 143.6455536
30: -83.3655701, 56.1972656, -83.1562729, 55.8417473, -139.2073212, 139.3535461
31: -90.0213470, 49.0936165, -89.7249832, 48.9385757, -138.9599304, 138.8185883
32: -76.9185333, 59.5698318, -76.6850357, 59.4444313, -136.3629608, 136.2548523
33: -113.4989929, 71.0707779, -113.1836166, 70.8574600, -184.3564453, 184.2543945
34: -90.2840118, 52.2858849, -90.1108322, 52.1634941, -142.4474945, 142.3967133
35: -88.9208832, 63.9394875, -88.6052856, 63.8014221, -152.7223053, 152.5447693
36: -87.9945068, 60.6889496, -87.7049332, 60.6041832, -148.5986938, 148.3938904
37: -136.4391327, 45.3963165, -136.1220093, 45.2931061, -181.7322235, 181.5183258
38: -109.7480774, 68.7172318, -109.4174957, 68.5761566, -178.3242188, 178.1347351
39: -122.1878662, 68.4695816, -121.7749939, 68.2163239, -190.4041748, 190.2445679
40: -112.8925247, 35.5733910, -112.7093735, 35.4191589, -148.3116760, 148.2827606
41: -86.2139130, 49.5339928, -85.9503326, 49.4644394, -135.6783447, 135.4843292
42: -58.3162575, 46.9217491, -58.1700592, 46.8491096, -105.1653671, 105.0918045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0872453, upper bound: 118.9101911
time: 137.64 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0872453, upper bound: 119.0754295
time: 122.21 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -103.3441925, 67.4190750, -103.4646759, 67.3620605, -170.7062531, 170.8837585
1: -49.8722191, 50.6513672, -49.9443512, 50.6645546, -100.5367737, 100.5957184
2: -48.5012360, 50.0659714, -48.5620193, 50.0418015, -98.5430145, 98.6279907
3: -52.0522003, 63.2399063, -52.1598434, 63.2294273, -115.2816315, 115.3997498
4: -65.5444336, 56.9830704, -65.6047058, 56.9559898, -122.5004120, 122.5877762
5: -55.0804749, 58.9813271, -55.2282372, 58.9532166, -114.0336914, 114.2095490
6: -86.3355408, 50.4855423, -86.3846588, 50.6175308, -136.9530640, 136.8701782
7: -67.1477356, 49.1696167, -67.3088608, 49.1555824, -116.3033142, 116.4784775
8: -78.6041794, 69.7411957, -78.7043228, 69.6910400, -148.2952271, 148.4455261
9: -61.5676575, 68.0215149, -61.6677933, 68.0651855, -129.6328430, 129.6893005
10: -91.4691391, 89.6189423, -91.6647797, 89.8141022, -181.2832336, 181.2837067
11: -83.4184952, 42.3302650, -83.5702972, 42.3971519, -125.8156433, 125.9005585
12: -60.3129272, 76.4154739, -60.3757553, 76.6608582, -136.9737701, 136.7912292
13: -67.8247910, 104.9750366, -67.9200287, 105.1247101, -172.9494934, 172.8950653
14: -116.7636871, 60.7188950, -116.9072800, 60.6877365, -177.4514160, 177.6261444
15: -66.7740784, 63.3985481, -66.9088745, 63.4485054, -130.2225647, 130.3074188
16: -97.3149185, 53.8959732, -97.4976883, 54.0535278, -151.3684387, 151.3936615
17: -109.5196228, 72.3545990, -109.7354736, 72.3768997, -181.8965149, 182.0900726
18: -91.0921326, 46.6164780, -91.2157516, 46.6953583, -137.7874756, 137.8322144
19: -67.0544357, 34.9847794, -67.1167374, 35.0300789, -102.0845184, 102.1015091
20: -65.6555634, 42.6235962, -65.8040771, 42.6665344, -108.3220825, 108.4276733
21: -84.8669739, 42.8316917, -85.0206375, 42.8745041, -127.7414703, 127.8523254
22: -74.9288330, 62.0914230, -75.1526642, 62.1375999, -137.0664215, 137.2440796
23: -66.9770355, 48.2241898, -67.0078125, 48.2608566, -115.2378922, 115.2320023
24: -85.6474152, 55.4893646, -85.8182983, 55.5488968, -141.1963043, 141.3076477
25: -63.0544930, 57.9007988, -63.1538200, 57.9373512, -120.9918442, 121.0546188
26: -90.6854630, 57.3528099, -90.8362274, 57.4259529, -148.1114197, 148.1890106
27: -99.0305328, 46.0381317, -99.2472382, 46.1022034, -145.1327362, 145.2853699
28: -66.6293716, 53.2709503, -66.6883087, 53.3137245, -119.9430923, 119.9592514
29: -77.9383850, 65.4631958, -78.2111359, 65.4977341, -143.4361267, 143.6743164
30: -83.0507507, 55.9075165, -83.2915878, 55.9512062, -139.0019531, 139.1990967
31: -89.7823410, 48.9699059, -89.8464355, 49.0272255, -138.8095703, 138.8163452
32: -76.7626495, 59.4337158, -76.8319092, 59.5413857, -136.3040161, 136.2656250
33: -113.2609253, 70.7924652, -113.3126526, 70.9612732, -184.2221680, 184.1051178
34: -90.1589508, 51.9286957, -90.2081604, 52.1932831, -142.3522339, 142.1368408
35: -88.6641388, 63.5780258, -88.7193375, 63.8602066, -152.5243225, 152.2973633
36: -87.7544861, 60.4036407, -87.8091583, 60.6219940, -148.3764648, 148.2127991
37: -136.1348877, 45.1396561, -136.2216187, 45.3467560, -181.4816284, 181.3612671
38: -109.5206375, 68.2685242, -109.5307922, 68.6478729, -178.1685181, 177.7993164
39: -121.9149933, 68.1627960, -121.9797821, 68.3768616, -190.2918396, 190.1425781
40: -112.7325745, 35.4230232, -112.8311539, 35.5006638, -148.2332153, 148.2541809
41: -86.0441437, 49.3402176, -86.0865326, 49.5468063, -135.5909424, 135.4267578
42: -58.2126350, 46.8446846, -58.2930031, 46.9834366, -105.1960754, 105.1376801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=604, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1569

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2349777, upper bound: 118.9622509
time: 140.96 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0876273, upper bound: 119.1221559
time: 101.60 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -103.5423813, 67.4998703, -103.4832687, 67.3676453, -170.9100342, 170.9831390
1: -49.9941978, 50.7524872, -49.9584770, 50.6720695, -100.6662521, 100.7109680
2: -48.5946045, 50.1276474, -48.5718994, 50.0468826, -98.6414871, 98.6995468
3: -52.1726189, 63.3310394, -52.1743927, 63.2332535, -115.4058685, 115.5054321
4: -65.6412048, 57.0749397, -65.6140747, 56.9629097, -122.6041031, 122.6890106
5: -55.2565308, 59.0919113, -55.2510910, 58.9600945, -114.2166061, 114.3430023
6: -86.4478912, 50.5869217, -86.3903809, 50.6279068, -137.0757904, 136.9772949
7: -67.3797607, 49.2912674, -67.3383026, 49.1618805, -116.5416412, 116.6295547
8: -78.7321014, 69.8794479, -78.7197723, 69.6988449, -148.4309387, 148.5992126
9: -61.7204323, 68.1216431, -61.6851921, 68.0735550, -129.7939758, 129.8068390
10: -91.7545547, 89.7395172, -91.7002106, 89.8237762, -181.5783386, 181.4397125
11: -83.7941132, 42.4880524, -83.6176910, 42.4055862, -126.1996918, 126.1057434
12: -60.5938148, 76.6885452, -60.3860970, 76.6969528, -137.2907410, 137.0746460
13: -68.1550446, 105.2621460, -67.9346466, 105.1617966, -173.3168030, 173.1967926
14: -116.9936218, 60.7822037, -116.9302902, 60.6928558, -177.6864624, 177.7124786
15: -66.8849258, 63.4889336, -66.9214706, 63.4535713, -130.3385010, 130.4104004
16: -97.5982590, 54.0253448, -97.5296631, 54.0628357, -151.6610870, 151.5550079
17: -109.8367996, 72.6314545, -109.7725372, 72.4131622, -182.2499695, 182.4039917
18: -91.2968597, 46.7525406, -91.2392426, 46.7076950, -138.0045471, 137.9917908
19: -67.2289886, 35.0593529, -67.1377106, 35.0353584, -102.2643433, 102.1970673
20: -65.8736496, 42.8241043, -65.8300323, 42.6737175, -108.5473633, 108.6541290
21: -85.1857681, 43.0510139, -85.0601273, 42.8811913, -128.0669556, 128.1111450
22: -75.1790161, 62.2259598, -75.1776581, 62.1444702, -137.3234711, 137.4036102
23: -67.1346512, 48.3475800, -67.0258408, 48.2679863, -115.4026337, 115.3734131
24: -85.8844376, 55.6862488, -85.8446655, 55.5566177, -141.4410400, 141.5309143
25: -63.2545166, 58.0556183, -63.1764030, 57.9440918, -121.1986084, 121.2320251
26: -90.8688431, 57.5068970, -90.8531647, 57.4369583, -148.3058014, 148.3600464
27: -99.3050766, 46.2952576, -99.2799454, 46.1117058, -145.4167786, 145.5751953
28: -66.7244873, 53.3708801, -66.6990204, 53.3220406, -120.0465240, 120.0698929
29: -78.2915573, 65.6325302, -78.2508316, 65.5039597, -143.7955170, 143.8833618
30: -83.3664246, 56.2003555, -83.3291168, 55.9601288, -139.3265381, 139.5294495
31: -90.0221024, 49.0944748, -89.8737335, 49.0343781, -139.0564880, 138.9682007
32: -76.9221573, 59.5703659, -76.8399200, 59.5573883, -136.4795532, 136.4102783
33: -113.5015793, 71.0716934, -113.3224869, 70.9994659, -184.5010223, 184.3941803
34: -90.2852859, 52.2865982, -90.2148056, 52.2437363, -142.5290222, 142.5013885
35: -88.9235687, 63.9398575, -88.7275696, 63.9133835, -152.8369446, 152.6674194
36: -87.9966736, 60.6892853, -87.8162537, 60.6622162, -148.6588745, 148.5055389
37: -136.4398499, 45.3972702, -136.2346802, 45.3820801, -181.8219299, 181.6319427
38: -109.7502060, 68.7178268, -109.5390091, 68.7115860, -178.4617920, 178.2568359
39: -122.1933746, 68.4703751, -121.9917755, 68.4187927, -190.6121674, 190.4621429
40: -112.8938751, 35.5740662, -112.8396149, 35.5197868, -148.4136658, 148.4136810
41: -86.2168121, 49.5347023, -86.0912399, 49.5709610, -135.7877808, 135.6259460
42: -58.3189011, 46.9223824, -58.3003883, 46.9918251, -105.3107147, 105.2227631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1569

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2349777, upper bound: 119.0752631
time: 166.72 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0876273, upper bound: 119.2349770
time: 144.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 313.85 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -118.9225054, upper bound: 119.1949155
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -119.0347361, upper bound: 119.1949155
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -119.0838677, upper bound: 119.1957043
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -119.1957038, upper bound: 119.1957043
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -118.8216133, upper bound: 119.2420110
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -118.9340658, upper bound: 119.2420110
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -118.9831332, upper bound: 119.2428069
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -119.0954490, upper bound: 119.2428068
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -119.0872453, upper bound: 118.7970483
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -119.0872453, upper bound: 118.9624394
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -119.0872453, upper bound: 118.9101911
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -119.0872453, upper bound: 119.0754295
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -119.2349777, upper bound: 118.9622509
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -119.0876273, upper bound: 119.1221559
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -119.2349777, upper bound: 119.0752631
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 313.85
Output dim: 13, lower bound: -119.0876273, upper bound: 119.2349770
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=172.8270263671875
rel_dist={13: [-119.26634416186121, 119.2663441598697]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 576

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1725

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8132670, upper bound: 115.9344275
time: 122.48 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8132670, upper bound: 115.9344275
time: 122.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 245.22 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 245.22
Output dim: 13, lower bound: -115.8132670, upper bound: 115.9344275
IS_A2, status: Status.UNKNOWN, split count: 1, time: 245.22
Output dim: 13, lower bound: -115.8132670, upper bound: 115.9344275

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -103.2789230, 67.2919617, -103.4047470, 67.3250580, -170.6039581, 170.6967163
1: -49.8466568, 50.5750504, -49.9255905, 50.6008453, -100.4475021, 100.5006409
2: -48.4513588, 49.9596710, -48.5355606, 49.9839478, -98.4352875, 98.4952240
3: -52.0105362, 63.1161232, -52.1183052, 63.1471405, -115.1576767, 115.2344284
4: -65.4317474, 56.8847351, -65.5585327, 56.9114647, -122.3432159, 122.4432602
5: -55.1246986, 58.8373146, -55.2117653, 58.8726425, -113.9973450, 114.0490799
6: -86.2812881, 50.4314461, -86.3399963, 50.4883423, -136.7696228, 136.7714386
7: -67.1984406, 49.0380058, -67.2989807, 49.0695190, -116.2679596, 116.3369904
8: -78.5159607, 69.6070328, -78.6490173, 69.6473465, -148.1632996, 148.2560425
9: -61.5627823, 67.9424744, -61.6399078, 67.9731827, -129.5359650, 129.5823669
10: -91.5939331, 89.5898285, -91.6570358, 89.6428604, -181.2367859, 181.2468567
11: -83.5207901, 42.2501907, -83.5579453, 42.3459015, -125.8666916, 125.8081360
12: -60.2727776, 76.4904327, -60.3410606, 76.5564270, -136.8292084, 136.8314972
13: -67.5933228, 104.8646698, -67.8439026, 104.9021072, -172.4954224, 172.7085419
14: -116.6980896, 60.6436386, -116.8134995, 60.6706238, -177.3687134, 177.4571381
15: -66.7402344, 63.3593292, -66.8108063, 63.4164505, -130.1566772, 130.1701355
16: -97.4204788, 53.8608360, -97.4820404, 53.8994522, -151.3199158, 151.3428802
17: -109.5036087, 72.3375397, -109.6601791, 72.3702850, -181.8738556, 181.9977112
18: -90.9898911, 46.4713249, -91.0318756, 46.6418991, -137.6317749, 137.5031891
19: -67.0177841, 34.8906326, -67.0492783, 34.9951668, -102.0129395, 101.9399109
20: -65.6934586, 42.5447807, -65.7245255, 42.6401749, -108.3336182, 108.2692947
21: -84.9193420, 42.7168999, -84.9560776, 42.8425293, -127.7618713, 127.6729736
22: -74.9042816, 62.0042725, -74.9493179, 62.1083221, -137.0126038, 136.9535828
23: -66.8766403, 48.0928650, -66.9039688, 48.2227859, -115.0994263, 114.9968338
24: -85.6265869, 55.3728523, -85.6635590, 55.5097275, -141.1363220, 141.0364075
25: -63.0124893, 57.7699051, -63.0474815, 57.9006500, -120.9131393, 120.8173828
26: -90.5337219, 57.2201385, -90.5889740, 57.3791313, -147.9128571, 147.8091125
27: -99.0048828, 45.8935585, -99.0427551, 46.0603943, -145.0652466, 144.9363098
28: -66.5087280, 53.1532402, -66.5378799, 53.2812614, -119.7899780, 119.6911163
29: -78.0079727, 65.3635406, -78.0496292, 65.4693832, -143.4773560, 143.4131622
30: -83.1591110, 55.8056870, -83.1927032, 55.9208183, -139.0799255, 138.9983826
31: -89.7233124, 48.8365593, -89.7655334, 48.9809647, -138.7042694, 138.6020966
32: -76.6843872, 59.4214249, -76.7705383, 59.4633636, -136.1477356, 136.1919556
33: -113.1405029, 70.8516159, -113.2579193, 70.8919830, -184.0324707, 184.1095276
34: -90.0883026, 52.1685448, -90.1613083, 52.1951294, -142.2834167, 142.3298492
35: -88.5957031, 63.7896461, -88.6773300, 63.8241043, -152.4197998, 152.4669800
36: -87.6762085, 60.5937653, -87.7599869, 60.6206589, -148.2968750, 148.3537598
37: -136.0542297, 45.2893219, -136.1515045, 45.3249435, -181.3791504, 181.4408264
38: -109.3837280, 68.5697327, -109.4801865, 68.6051178, -177.9888458, 178.0499268
39: -121.7112503, 68.2278595, -121.8982239, 68.2489929, -189.9602356, 190.1260681
40: -112.6575012, 35.4271545, -112.7446823, 35.4557152, -148.1132202, 148.1718445
41: -85.9746704, 49.4464340, -86.0363312, 49.4863548, -135.4610291, 135.4827576
42: -58.2059669, 46.8197632, -58.2506142, 46.8718529, -105.0778198, 105.0703735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=461, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1592

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8058997, upper bound: 115.7994666
time: 139.08 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8064082, upper bound: 115.9276136
time: 112.46 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -103.4987564, 67.4807739, -103.4331207, 67.3322372, -170.8309784, 170.9138947
1: -49.9841461, 50.7235031, -49.9433441, 50.6062241, -100.5903625, 100.6668472
2: -48.5848351, 50.1187897, -48.5574493, 49.9890518, -98.5738831, 98.6762390
3: -52.1724319, 63.3241463, -52.1437416, 63.1537819, -115.3262024, 115.4678802
4: -65.6181717, 57.0589294, -65.5852203, 56.9168320, -122.5350037, 122.6441345
5: -55.2560272, 59.0489693, -55.2306976, 58.8801537, -114.1361847, 114.2796631
6: -86.4045105, 50.5677490, -86.3521729, 50.5014229, -136.9059296, 136.9199219
7: -67.3689041, 49.2245445, -67.3210831, 49.0764084, -116.4452972, 116.5456238
8: -78.7240067, 69.8168335, -78.6795197, 69.6562271, -148.3802338, 148.4963379
9: -61.7092667, 68.0905609, -61.6555634, 67.9791718, -129.6884460, 129.7461243
10: -91.7617569, 89.7050629, -91.6712952, 89.6536255, -181.4153748, 181.3763580
11: -83.7840271, 42.4016991, -83.5649261, 42.3683929, -126.1524200, 125.9666290
12: -60.4028091, 76.6851578, -60.3556366, 76.5708313, -136.9736328, 137.0407715
13: -67.9509506, 105.2626648, -67.9037094, 104.9096069, -172.8605652, 173.1663666
14: -116.9430313, 60.7728233, -116.8380585, 60.6761398, -177.6191711, 177.6108856
15: -66.8750763, 63.4576836, -66.8266754, 63.4269333, -130.3020020, 130.2843628
16: -97.5842438, 53.9723434, -97.4953537, 53.9073410, -151.4915771, 151.4676971
17: -109.7997818, 72.6318970, -109.6957092, 72.3772430, -182.1770172, 182.3276062
18: -91.2873077, 46.7206841, -91.0405426, 46.6822815, -137.9695892, 137.7612305
19: -67.2232437, 35.0353203, -67.0552673, 35.0199051, -102.2431412, 102.0905914
20: -65.8555374, 42.6894531, -65.7301941, 42.6622314, -108.5177612, 108.4196472
21: -85.1621933, 42.8966904, -84.9623871, 42.8720779, -128.0342712, 127.8590775
22: -75.1213989, 62.1596489, -74.9584198, 62.1325989, -137.2539978, 137.1180725
23: -67.1238098, 48.2861977, -66.9084473, 48.2535858, -115.3773956, 115.1946335
24: -85.8645935, 55.5689545, -85.6702881, 55.5423431, -141.4069366, 141.2392426
25: -63.2279701, 57.9655228, -63.0541458, 57.9313164, -121.1592865, 121.0196609
26: -90.8253021, 57.4471397, -90.6002197, 57.4163170, -148.2416077, 148.0473633
27: -99.2961731, 46.1256485, -99.0495758, 46.1001205, -145.3963013, 145.1752319
28: -66.7207642, 53.3386803, -66.5432053, 53.3114052, -120.0321503, 119.8818741
29: -78.2435379, 65.5223541, -78.0576553, 65.4951172, -143.7386475, 143.5800171
30: -83.3387451, 55.9877357, -83.1987915, 55.9477463, -139.2864990, 139.1865234
31: -90.0051270, 49.0414886, -89.7736969, 49.0150261, -139.0201416, 138.8151855
32: -76.8434982, 59.5563354, -76.7890320, 59.4728470, -136.3163452, 136.3453674
33: -113.3452225, 71.0796661, -113.2836533, 70.9006042, -184.2457886, 184.3633118
34: -90.2239685, 52.2682648, -90.1754532, 52.2002106, -142.4241638, 142.4437256
35: -88.7385101, 63.9412918, -88.6927948, 63.8317795, -152.5702820, 152.6340942
36: -87.8224945, 60.6886368, -87.7764587, 60.6264343, -148.4489136, 148.4650879
37: -136.2642822, 45.4085770, -136.1720276, 45.3325729, -181.5968628, 181.5805969
38: -109.5997543, 68.6971359, -109.5018539, 68.6128693, -178.2126160, 178.1989899
39: -122.0251160, 68.4777374, -121.9437485, 68.2532196, -190.2783051, 190.4214783
40: -112.8514175, 35.5672913, -112.7634583, 35.4612122, -148.3126221, 148.3307495
41: -86.1057053, 49.5208855, -86.0496674, 49.4943504, -135.6000519, 135.5705566
42: -58.3013573, 46.9149704, -58.2601700, 46.8822327, -105.1835938, 105.1751404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=461, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 576

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1592

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8058997, upper bound: 115.7994666
time: 104.98 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8064082, upper bound: 115.9276136
time: 108.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 216.25 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 216.25
Output dim: 13, lower bound: -115.8058997, upper bound: 115.7994666
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 216.25
Output dim: 13, lower bound: -115.8064082, upper bound: 115.9276136
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 216.25
Output dim: 13, lower bound: -115.8058997, upper bound: 115.7994666
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 216.25
Output dim: 13, lower bound: -115.8064082, upper bound: 115.9276136

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -103.2654877, 67.2852783, -103.3377075, 67.2915726, -170.5570679, 170.6229858
1: -49.8334389, 50.5711594, -49.8597069, 50.5813065, -100.4147415, 100.4308624
2: -48.4396706, 49.9550247, -48.4772263, 49.9604568, -98.4001160, 98.4322510
3: -51.9930534, 63.1120682, -52.0313454, 63.1266937, -115.1197510, 115.1434174
4: -65.4234161, 56.8795776, -65.5167236, 56.8856697, -122.3090820, 122.3963013
5: -55.1103935, 58.8329620, -55.1411324, 58.8506889, -113.9610825, 113.9740829
6: -86.2652435, 50.4254303, -86.2588806, 50.4582710, -136.7235107, 136.6842957
7: -67.1777802, 49.0340691, -67.1967773, 49.0497665, -116.2275391, 116.2308426
8: -78.5086288, 69.5991592, -78.6123352, 69.6076736, -148.1163025, 148.2114868
9: -61.5469971, 67.9386902, -61.5624847, 67.9542542, -129.5012512, 129.5011749
10: -91.5672607, 89.5823059, -91.5222397, 89.6054382, -181.1726837, 181.1045380
11: -83.5071793, 42.2443466, -83.4898987, 42.3164330, -125.8236084, 125.7342453
12: -60.2574921, 76.4809799, -60.2636795, 76.5098953, -136.7673950, 136.7446594
13: -67.5497513, 104.8579025, -67.6229401, 104.8683014, -172.4180450, 172.4808350
14: -116.6860504, 60.6249504, -116.7534485, 60.5760193, -177.2620392, 177.3784027
15: -66.7317200, 63.3403435, -66.7684174, 63.3208466, -130.0525665, 130.1087646
16: -97.3985367, 53.8539429, -97.3716736, 53.8645897, -151.2631226, 151.2256165
17: -109.4897995, 72.3244629, -109.5914230, 72.3046570, -181.7944489, 181.9158936
18: -90.9829407, 46.4254150, -90.9971008, 46.4169998, -137.3999329, 137.4225159
19: -67.0109253, 34.8778152, -67.0151062, 34.9302025, -101.9411087, 101.8929214
20: -65.6871872, 42.5260162, -65.6933670, 42.5452271, -108.2324066, 108.2193832
21: -84.9101257, 42.6998634, -84.9100113, 42.7564621, -127.6665878, 127.6098633
22: -74.8961182, 61.9718704, -74.9082642, 61.9439240, -136.8400421, 136.8801270
23: -66.8713989, 48.0768433, -66.8778381, 48.1418571, -115.0132523, 114.9546814
24: -85.6207123, 55.3471298, -85.6339417, 55.3806229, -141.0013123, 140.9810638
25: -63.0068436, 57.7498245, -63.0191612, 57.8002777, -120.8071213, 120.7689819
26: -90.5260773, 57.1764488, -90.5506439, 57.1579247, -147.6839905, 147.7270966
27: -98.9980774, 45.8563538, -99.0087891, 45.8716431, -144.8697205, 144.8651428
28: -66.5032806, 53.1282082, -66.5108566, 53.1543198, -119.6576004, 119.6390533
29: -78.0006104, 65.3376160, -78.0126495, 65.3379822, -143.3385925, 143.3502655
30: -83.1511536, 55.7846909, -83.1528778, 55.8162079, -138.9673462, 138.9375610
31: -89.7146988, 48.8213387, -89.7223969, 48.9052505, -138.6199493, 138.5437317
32: -76.6637192, 59.4160004, -76.6682587, 59.4364662, -136.1001740, 136.0842590
33: -113.1206894, 70.8435516, -113.1579895, 70.8519287, -183.9726257, 184.0015411
34: -90.0756073, 52.1618614, -90.0973663, 52.1618538, -142.2374573, 142.2592163
35: -88.5781555, 63.7841721, -88.5899353, 63.7968483, -152.3750000, 152.3740845
36: -87.6620483, 60.5901527, -87.6887512, 60.6025391, -148.2645874, 148.2789001
37: -136.0449982, 45.2819214, -136.1052704, 45.2880287, -181.3330231, 181.3871918
38: -109.3673096, 68.5630951, -109.3974686, 68.5723419, -177.9396515, 177.9605713
39: -121.6783142, 68.2210464, -121.7312317, 68.2152557, -189.8935699, 189.9522705
40: -112.6481705, 35.4188843, -112.6977844, 35.4146500, -148.0628052, 148.1166687
41: -85.9555206, 49.4406929, -85.9396667, 49.4575958, -135.4131165, 135.3803558
42: -58.1889534, 46.8132439, -58.1650505, 46.8393021, -105.0282440, 104.9782867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 693

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8029113, upper bound: 115.6916929
time: 122.05 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8029113, upper bound: 115.7964858
time: 93.49 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -103.2724686, 67.2901306, -103.4564209, 67.3631592, -170.6356201, 170.7465515
1: -49.8434601, 50.5738792, -49.9411240, 50.6690369, -100.5124817, 100.5149994
2: -48.4482002, 49.9585114, -48.5516777, 50.0439148, -98.4921112, 98.5101852
3: -52.0057755, 63.1143875, -52.1503983, 63.2345772, -115.2403412, 115.2647858
4: -65.4297638, 56.8832550, -65.5864639, 56.9610786, -122.3908234, 122.4697189
5: -55.1204376, 58.8359528, -55.2328873, 58.9548912, -114.0753250, 114.0688324
6: -86.2779083, 50.4296875, -86.3808441, 50.6161270, -136.8940125, 136.8105316
7: -67.1932983, 49.0368729, -67.3166656, 49.1562157, -116.3495026, 116.3535385
8: -78.5143127, 69.6046295, -78.6901550, 69.6921844, -148.2064972, 148.2947693
9: -61.5597801, 67.9410629, -61.6722336, 68.0685883, -129.6283722, 129.6132965
10: -91.5886841, 89.5877686, -91.6937866, 89.8147583, -181.4034119, 181.2815552
11: -83.5169830, 42.2483597, -83.6155701, 42.3843613, -125.9013443, 125.8639297
12: -60.2694778, 76.4884644, -60.3725204, 76.6846008, -136.9540710, 136.8609924
13: -67.5855713, 104.8624115, -67.8749390, 105.1577301, -172.7433014, 172.7373352
14: -116.6948395, 60.6378365, -116.9069443, 60.6893883, -177.3842316, 177.5447693
15: -66.7379761, 63.3556061, -66.9068451, 63.4443932, -130.1823730, 130.2624512
16: -97.4118423, 53.8589745, -97.5227203, 54.0558090, -151.4676514, 151.3816986
17: -109.4991684, 72.3320618, -109.7387924, 72.4123535, -181.9115295, 182.0708618
18: -90.9873581, 46.4628143, -91.2343826, 46.6685562, -137.6558838, 137.6971893
19: -67.0158463, 34.8882065, -67.1354828, 35.0110474, -102.0268936, 102.0236664
20: -65.6918335, 42.5411987, -65.8259430, 42.6526146, -108.3444290, 108.3671417
21: -84.9167633, 42.7134781, -85.0572052, 42.8526039, -127.7693634, 127.7706757
22: -74.9018402, 61.9987373, -75.1708450, 62.1212387, -137.0230713, 137.1695709
23: -66.8748398, 48.0896912, -67.0235062, 48.2372742, -115.1121140, 115.1131973
24: -85.6241608, 55.3682518, -85.8414764, 55.5256882, -141.1498413, 141.2097168
25: -63.0108910, 57.7657776, -63.1715851, 57.9138184, -120.9247131, 120.9373627
26: -90.5312958, 57.2124939, -90.8438339, 57.4001884, -147.9314880, 148.0563354
27: -99.0027618, 45.8867226, -99.2772675, 46.0726509, -145.0754089, 145.1639709
28: -66.5073242, 53.1485863, -66.6951752, 53.2914925, -119.7988129, 119.8437500
29: -78.0052490, 65.3586578, -78.2462463, 65.4786530, -143.4838867, 143.6048889
30: -83.1569061, 55.8012123, -83.3256836, 55.9345779, -139.0914917, 139.1268921
31: -89.7207413, 48.8311920, -89.8711700, 49.0010490, -138.7217865, 138.7023621
32: -76.6808777, 59.4198761, -76.8231201, 59.5494423, -136.2303162, 136.2429962
33: -113.1357040, 70.8494415, -113.2968750, 70.9938812, -184.1295471, 184.1463013
34: -90.0846863, 52.1666603, -90.2012939, 52.2421074, -142.3267975, 142.3679504
35: -88.5920258, 63.7877388, -88.7121735, 63.9088364, -152.5008545, 152.4999084
36: -87.6732635, 60.5927734, -87.8000870, 60.6606522, -148.3339233, 148.3928528
37: -136.0512695, 45.2875290, -136.2179565, 45.3769951, -181.4282532, 181.5054932
38: -109.3798218, 68.5678024, -109.5189972, 68.7077713, -178.0875854, 178.0867920
39: -121.7052383, 68.2260742, -121.9480362, 68.4177246, -190.1229553, 190.1741028
40: -112.6554413, 35.4245148, -112.8280640, 35.5152740, -148.1707153, 148.2525787
41: -85.9706879, 49.4450340, -86.0806046, 49.5641327, -135.5348206, 135.5256348
42: -58.2025261, 46.8178596, -58.2953720, 46.9819870, -105.1845016, 105.1132278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 693

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8034167, upper bound: 115.8199144
time: 126.49 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8034167, upper bound: 115.9245929
time: 92.23 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -103.4853058, 67.4741058, -103.3660965, 67.2987671, -170.7840729, 170.8401947
1: -49.9709511, 50.7196274, -49.8774529, 50.5866814, -100.5576324, 100.5970764
2: -48.5731354, 50.1141357, -48.4991302, 49.9655342, -98.5386581, 98.6132584
3: -52.1549072, 63.3200684, -52.0567513, 63.1333389, -115.2882462, 115.3768158
4: -65.6098328, 57.0537910, -65.5434341, 56.8910637, -122.5009003, 122.5972137
5: -55.2417259, 59.0446167, -55.1600304, 58.8582191, -114.0999451, 114.2046509
6: -86.3884888, 50.5617180, -86.2709961, 50.4713669, -136.8598328, 136.8327026
7: -67.3482666, 49.2206573, -67.2188950, 49.0566330, -116.4048996, 116.4395523
8: -78.7166367, 69.8089600, -78.6427917, 69.6165771, -148.3332214, 148.4517365
9: -61.6935005, 68.0867920, -61.5781250, 67.9602051, -129.6537018, 129.6649170
10: -91.7350769, 89.6975479, -91.5364914, 89.6161575, -181.3512268, 181.2340393
11: -83.7704239, 42.3958511, -83.4969101, 42.3389206, -126.1093445, 125.8927536
12: -60.3875313, 76.6757812, -60.2782860, 76.5243149, -136.9118500, 136.9540558
13: -67.9073334, 105.2559128, -67.6827316, 104.8758545, -172.7831879, 172.9386444
14: -116.9309921, 60.7541122, -116.7779007, 60.5815315, -177.5125275, 177.5320129
15: -66.8666077, 63.4386864, -66.7843323, 63.3313828, -130.1979980, 130.2230225
16: -97.5622711, 53.9654236, -97.3849945, 53.8724480, -151.4347076, 151.3504181
17: -109.7859879, 72.6188202, -109.6269455, 72.3116074, -182.0975952, 182.2457428
18: -91.2803268, 46.6747971, -91.0057831, 46.4573174, -137.7376404, 137.6805725
19: -67.2164154, 35.0224724, -67.0210876, 34.9549408, -102.1713562, 102.0435638
20: -65.8492737, 42.6707077, -65.6990356, 42.5672836, -108.4165573, 108.3697433
21: -85.1530151, 42.8796463, -84.9163284, 42.7860374, -127.9390564, 127.7959747
22: -75.1132202, 62.1272583, -74.9173737, 61.9681892, -137.0814056, 137.0446320
23: -67.1185303, 48.2701797, -66.8823242, 48.1726608, -115.2911911, 115.1525040
24: -85.8586731, 55.5432625, -85.6407013, 55.4132118, -141.2718811, 141.1839600
25: -63.2223129, 57.9454498, -63.0258331, 57.8309669, -121.0532684, 120.9712830
26: -90.8176270, 57.4034195, -90.5618668, 57.1951294, -148.0127411, 147.9652863
27: -99.2893982, 46.0884476, -99.0155792, 45.9114304, -145.2008362, 145.1040344
28: -66.7152863, 53.3136444, -66.5161743, 53.1844940, -119.8997803, 119.8298187
29: -78.2361908, 65.4964828, -78.0206909, 65.3637009, -143.5998840, 143.5171661
30: -83.3307648, 55.9667320, -83.1589508, 55.8431358, -139.1739044, 139.1256714
31: -89.9965210, 49.0262108, -89.7305756, 48.9392853, -138.9358063, 138.7567749
32: -76.8228455, 59.5509567, -76.6867218, 59.4459534, -136.2687836, 136.2376556
33: -113.3254166, 71.0717239, -113.1836700, 70.8605042, -184.1859131, 184.2554016
34: -90.2112274, 52.2615776, -90.1115036, 52.1669579, -142.3781891, 142.3730621
35: -88.7209473, 63.9358330, -88.6053925, 63.8045197, -152.5254517, 152.5412140
36: -87.8083496, 60.6849823, -87.7052002, 60.6083260, -148.4166718, 148.3901825
37: -136.2550507, 45.4012146, -136.1257629, 45.2957726, -181.5508118, 181.5269775
38: -109.5833588, 68.6905289, -109.4191589, 68.5800171, -178.1633606, 178.1096802
39: -121.9921188, 68.4708862, -121.7766724, 68.2194290, -190.2115326, 190.2475586
40: -112.8420792, 35.5590172, -112.7165451, 35.4201889, -148.2622681, 148.2755585
41: -86.0865784, 49.5151596, -85.9530182, 49.4655609, -135.5521393, 135.4681702
42: -58.2843590, 46.9084320, -58.1746292, 46.8497162, -105.1340637, 105.0830612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 693

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8029113, upper bound: 115.6916929
time: 93.74 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8029113, upper bound: 115.7964858
time: 92.58 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -103.4923096, 67.4789581, -103.4847946, 67.3703308, -170.8626404, 170.9637451
1: -49.9809723, 50.7223625, -49.9588203, 50.6743927, -100.6553574, 100.6811752
2: -48.5816689, 50.1176224, -48.5735550, 50.0490150, -98.6306839, 98.6911774
3: -52.1676712, 63.3223953, -52.1758347, 63.2412033, -115.4088745, 115.4982224
4: -65.6161804, 57.0574493, -65.6131897, 56.9664268, -122.5826111, 122.6706314
5: -55.2517395, 59.0476151, -55.2517891, 58.9623947, -114.2141342, 114.2994003
6: -86.4011536, 50.5659828, -86.3929749, 50.6292038, -137.0303650, 136.9589539
7: -67.3637695, 49.2234383, -67.3387909, 49.1630707, -116.5268402, 116.5622253
8: -78.7223282, 69.8144302, -78.7206116, 69.7010651, -148.4234009, 148.5350342
9: -61.7062263, 68.0891571, -61.6879196, 68.0745621, -129.7807922, 129.7770691
10: -91.7564774, 89.7030640, -91.7080231, 89.8254395, -181.5819092, 181.4110870
11: -83.7802582, 42.3998642, -83.6225739, 42.4068298, -126.1870880, 126.0224380
12: -60.3995552, 76.6832428, -60.3871498, 76.6990051, -137.0985565, 137.0703888
13: -67.9431839, 105.2604523, -67.9347076, 105.1652603, -173.1084442, 173.1951599
14: -116.9397812, 60.7669716, -116.9314117, 60.6949081, -177.6346893, 177.6983795
15: -66.8728561, 63.4539642, -66.9227142, 63.4548950, -130.3277588, 130.3766785
16: -97.5755615, 53.9704514, -97.5360336, 54.0636711, -151.6392212, 151.5064850
17: -109.7953186, 72.6264191, -109.7742157, 72.4193268, -182.2146454, 182.4006042
18: -91.2847290, 46.7122040, -91.2430573, 46.7089233, -137.9936523, 137.9552460
19: -67.2213058, 35.0328407, -67.1414261, 35.0357704, -102.2570724, 102.1742706
20: -65.8539200, 42.6858673, -65.8316269, 42.6746521, -108.5285721, 108.5174942
21: -85.1596222, 42.8932571, -85.0634842, 42.8821449, -128.0417633, 127.9567413
22: -75.1189651, 62.1541100, -75.1798706, 62.1455421, -137.2645111, 137.3339844
23: -67.1220016, 48.2830544, -67.0279694, 48.2680511, -115.3900452, 115.3110199
24: -85.8621597, 55.5644035, -85.8482056, 55.5582504, -141.4204102, 141.4126129
25: -63.2263603, 57.9613647, -63.1782761, 57.9445343, -121.1708984, 121.1396408
26: -90.8228912, 57.4394913, -90.8550415, 57.4374123, -148.2602997, 148.2945251
27: -99.2940826, 46.1187744, -99.2840500, 46.1124077, -145.4064789, 145.4028320
28: -66.7193756, 53.3340378, -66.7005005, 53.3216667, -120.0410080, 120.0345230
29: -78.2408600, 65.5175018, -78.2542648, 65.5043716, -143.7452393, 143.7717590
30: -83.3365326, 55.9832802, -83.3317566, 55.9615479, -139.2980499, 139.3150330
31: -90.0025635, 49.0360870, -89.8793106, 49.0350456, -139.0376129, 138.9154053
32: -76.8399963, 59.5548172, -76.8416138, 59.5589218, -136.3989258, 136.3964233
33: -113.3404388, 71.0774994, -113.3225479, 71.0024872, -184.3429260, 184.4000397
34: -90.2203217, 52.2663574, -90.2154694, 52.2472343, -142.4675446, 142.4818268
35: -88.7348404, 63.9393921, -88.7276917, 63.9164658, -152.6513062, 152.6670837
36: -87.8195724, 60.6875839, -87.8165817, 60.6663895, -148.4859314, 148.5041656
37: -136.2613525, 45.4068375, -136.2384491, 45.3847122, -181.6460571, 181.6452789
38: -109.5958633, 68.6951904, -109.5406723, 68.7154694, -178.3113403, 178.2358704
39: -122.0191193, 68.4758911, -121.9934921, 68.4219055, -190.4410095, 190.4693909
40: -112.8493500, 35.5646553, -112.8467865, 35.5208130, -148.3701477, 148.4114380
41: -86.1017456, 49.5194931, -86.0939484, 49.5721016, -135.6738434, 135.6134338
42: -58.2979355, 46.9130745, -58.3049507, 46.9923859, -105.2903137, 105.2180176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 693

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.9245932, upper bound: 115.8199144
time: 851.49 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8034167, upper bound: 115.9245929
time: 157.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 1011.33 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1011.33
Output dim: 13, lower bound: -115.8029113, upper bound: 115.6916929
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1011.33
Output dim: 13, lower bound: -115.8029113, upper bound: 115.7964858
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1011.33
Output dim: 13, lower bound: -115.8034167, upper bound: 115.8199144
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1011.33
Output dim: 13, lower bound: -115.8034167, upper bound: 115.9245929
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1011.33
Output dim: 13, lower bound: -115.8029113, upper bound: 115.6916929
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1011.33
Output dim: 13, lower bound: -115.8029113, upper bound: 115.7964858
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1011.33
Output dim: 13, lower bound: -115.9245932, upper bound: 115.8199144
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1011.33
Output dim: 13, lower bound: -115.8034167, upper bound: 115.9245929

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -103.1149521, 67.2248230, -103.2948456, 67.2746429, -170.3895874, 170.5196533
1: -49.7234154, 50.4997940, -49.8294258, 50.5616913, -100.2851028, 100.3292236
2: -48.3579941, 49.9030037, -48.4533043, 49.9458160, -98.3038025, 98.3563004
3: -51.8758316, 63.0290833, -51.9985237, 63.1033554, -114.9791794, 115.0275955
4: -65.3509064, 56.8047295, -65.4951324, 56.8646736, -122.2155762, 122.2998657
5: -54.9376450, 58.7662926, -55.0934296, 58.8319855, -113.7696304, 113.8597183
6: -86.1984863, 50.3443794, -86.2401276, 50.4346161, -136.6331024, 136.5845032
7: -66.9598236, 48.9799309, -67.1367493, 49.0344658, -115.9942780, 116.1166840
8: -78.3898697, 69.5252151, -78.5773010, 69.5873108, -147.9771729, 148.1025085
9: -61.4073792, 67.8707581, -61.5209465, 67.9349289, -129.3423157, 129.3916931
10: -91.2781296, 89.4974899, -91.4394608, 89.5814819, -180.8596191, 180.9369507
11: -83.1442108, 42.1740036, -83.3908310, 42.2955551, -125.4397583, 125.5648346
12: -60.1696777, 76.2125397, -60.2391357, 76.4359665, -136.6056519, 136.4516754
13: -67.4284515, 104.5716553, -67.5883942, 104.7906036, -172.2190552, 172.1600494
14: -116.5085220, 60.5747681, -116.7027893, 60.5612411, -177.0697174, 177.2775574
15: -66.6322021, 63.2836037, -66.7399902, 63.3048477, -129.9370422, 130.0235901
16: -97.1350098, 53.7789078, -97.2974396, 53.8437004, -150.9786987, 151.0763397
17: -109.2122650, 72.0508423, -109.5135956, 72.2255173, -181.4377747, 181.5644379
18: -90.7895050, 46.3266296, -90.9441376, 46.3867912, -137.1762695, 137.2707672
19: -66.8434143, 34.8287964, -66.9670944, 34.9165726, -101.7599869, 101.7958908
20: -65.4882660, 42.4623566, -65.6398315, 42.5269699, -108.0152206, 108.1021652
21: -84.6165848, 42.6370544, -84.8272858, 42.7385292, -127.3551102, 127.4643326
22: -74.7051697, 61.9071503, -74.8551178, 61.9258690, -136.6310425, 136.7622681
23: -66.7258606, 48.0167847, -66.8381042, 48.1246376, -114.8504944, 114.8548737
24: -85.4050980, 55.2705002, -85.5755310, 55.3592567, -140.7643585, 140.8460236
25: -62.8344536, 57.6877785, -62.9713669, 57.7829208, -120.6173553, 120.6591492
26: -90.3878174, 57.0870209, -90.5125351, 57.1315765, -147.5193939, 147.5995483
27: -98.7338257, 45.7732849, -98.9375000, 45.8479385, -144.5817413, 144.7107849
28: -66.4127960, 53.0634689, -66.4862061, 53.1357040, -119.5484848, 119.5496674
29: -77.6973190, 65.2815247, -77.9296722, 65.3222275, -143.0195465, 143.2111969
30: -82.8645935, 55.7073021, -83.0755310, 55.7932320, -138.6578064, 138.7828064
31: -89.4936295, 48.7531853, -89.6590576, 48.8863678, -138.3799744, 138.4122467
32: -76.5851746, 59.2944412, -76.6461563, 59.4022865, -135.9874573, 135.9405975
33: -113.0392685, 70.5577393, -113.1342926, 70.7732620, -183.8125305, 183.6920166
34: -90.0130234, 51.8236046, -90.0793304, 52.0619621, -142.0749817, 141.9029388
35: -88.5060883, 63.4221306, -88.5692902, 63.6915283, -152.1976013, 151.9914246
36: -87.5959930, 60.3058701, -87.6699219, 60.5196609, -148.1156616, 147.9757996
37: -135.9176025, 45.0141525, -136.0691986, 45.2158279, -181.1334229, 181.0833435
38: -109.2908554, 68.1359558, -109.3753815, 68.4468689, -177.7377319, 177.5113220
39: -121.5718536, 67.9073105, -121.6998749, 68.1304626, -189.7023163, 189.6071777
40: -112.5306473, 35.2763710, -112.6645432, 35.3754425, -147.9060822, 147.9409027
41: -85.8965378, 49.2608643, -85.9230347, 49.4081039, -135.3046265, 135.1838989
42: -58.1024551, 46.7442017, -58.1407547, 46.8197212, -104.9221802, 104.8849487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=604, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7979509, upper bound: 115.5525535
time: 96.07 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7979509, upper bound: 115.6858120
time: 95.93 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -103.3132172, 67.3055878, -103.3295212, 67.2853394, -170.5985565, 170.6351013
1: -49.8454590, 50.6009140, -49.8558311, 50.5758171, -100.4212799, 100.4567413
2: -48.4514351, 49.9646530, -48.4717369, 49.9554253, -98.4068451, 98.4363861
3: -51.9962845, 63.1201630, -52.0254822, 63.1106453, -115.1069183, 115.1456451
4: -65.4477234, 56.8965530, -65.5126724, 56.8777046, -122.3254242, 122.4092178
5: -55.1136742, 58.8768578, -55.1361122, 58.8448334, -113.9585114, 114.0129623
6: -86.3107758, 50.4457855, -86.2509460, 50.4538727, -136.7646484, 136.6967163
7: -67.1918945, 49.1015701, -67.1916809, 49.0461807, -116.2380676, 116.2932510
8: -78.5177917, 69.6634064, -78.6060867, 69.6018677, -148.1196594, 148.2695007
9: -61.5600967, 67.9707336, -61.5534210, 67.9505157, -129.5106201, 129.5241547
10: -91.5635223, 89.6180420, -91.5056839, 89.5994263, -181.1629028, 181.1237183
11: -83.5197601, 42.3318138, -83.4786072, 42.3113480, -125.8311005, 125.8104248
12: -60.4506531, 76.4856415, -60.2585144, 76.5033340, -136.9539642, 136.7441406
13: -67.7587738, 104.8588562, -67.6156616, 104.8599548, -172.6187286, 172.4745178
14: -116.7384644, 60.6380615, -116.7456512, 60.5709076, -177.3093719, 177.3837128
15: -66.7430496, 63.3739548, -66.7635040, 63.3143311, -130.0573730, 130.1374512
16: -97.4183273, 53.9083099, -97.3572235, 53.8610077, -151.2793274, 151.2655334
17: -109.5294876, 72.3276978, -109.5826035, 72.2924347, -181.8218994, 181.9102936
18: -90.9941864, 46.4627151, -90.9880829, 46.4096832, -137.4038696, 137.4508057
19: -67.0179443, 34.9034309, -67.0063324, 34.9264832, -101.9444122, 101.9097443
20: -65.7063675, 42.6629639, -65.6883774, 42.5403404, -108.2467041, 108.3513412
21: -84.9353790, 42.8563995, -84.9011078, 42.7510262, -127.6863861, 127.7575073
22: -74.9553223, 62.0417252, -74.9017029, 61.9386177, -136.8939362, 136.9434204
23: -66.8833771, 48.1402054, -66.8717270, 48.1379890, -115.0213623, 115.0119247
24: -85.6421356, 55.4673462, -85.6249619, 55.3736420, -141.0157776, 141.0923157
25: -63.0344429, 57.8426170, -63.0135574, 57.7955132, -120.8299561, 120.8561707
26: -90.5712280, 57.2411613, -90.5441437, 57.1522026, -147.7234344, 147.7853088
27: -99.0083771, 46.0304337, -98.9990616, 45.8656654, -144.8740387, 145.0294952
28: -66.5079193, 53.1633682, -66.5062256, 53.1511612, -119.6590805, 119.6695862
29: -78.0503845, 65.4509125, -78.0037766, 65.3339157, -143.3843079, 143.4546814
30: -83.1802521, 56.0002060, -83.1456909, 55.8099594, -138.9902039, 139.1459045
31: -89.7334213, 48.8777962, -89.7098312, 48.8996964, -138.6331177, 138.5876160
32: -76.7446442, 59.4310608, -76.6612167, 59.4321098, -136.1767426, 136.0922546
33: -113.2799225, 70.8370132, -113.1527863, 70.8445358, -184.1244507, 183.9898071
34: -90.1392212, 52.1813660, -90.0918045, 52.1545143, -142.2937164, 142.2731628
35: -88.7654877, 63.7839317, -88.5846405, 63.7899513, -152.5554199, 152.3685760
36: -87.8380966, 60.5915146, -87.6831818, 60.5944786, -148.4325714, 148.2746887
37: -136.2225647, 45.2717590, -136.0935974, 45.2816849, -181.5042419, 181.3653564
38: -109.5203323, 68.5851669, -109.3906708, 68.5637512, -178.0840759, 177.9758148
39: -121.8502808, 68.2148590, -121.7223969, 68.2083740, -190.0586548, 189.9372559
40: -112.6919403, 35.4272919, -112.6806946, 35.4110947, -148.1030273, 148.1079865
41: -86.0692291, 49.4553795, -85.9318771, 49.4532394, -135.5224609, 135.3872528
42: -58.2087059, 46.8218384, -58.1546669, 46.8352966, -105.0440063, 104.9765015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1569

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7979509, upper bound: 115.6574372
time: 93.75 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7979509, upper bound: 115.7907265
time: 138.35 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -103.1219559, 67.2296371, -103.4135437, 67.3462372, -170.4682007, 170.6431885
1: -49.7334290, 50.5024986, -49.9108467, 50.6493988, -100.3828278, 100.4133453
2: -48.3665581, 49.9065018, -48.5277519, 50.0292816, -98.3958435, 98.4342499
3: -51.8885765, 63.0313950, -52.1175842, 63.2112122, -115.0997925, 115.1489792
4: -65.3572235, 56.8084106, -65.5648499, 56.9400635, -122.2972794, 122.3732605
5: -54.9476318, 58.7693138, -55.1852036, 58.9361877, -113.8838043, 113.9545135
6: -86.2111511, 50.3486328, -86.3620987, 50.5925179, -136.8036652, 136.7107239
7: -66.9753647, 48.9827194, -67.2566833, 49.1408768, -116.1162262, 116.2394028
8: -78.3955231, 69.5306549, -78.6551132, 69.6717911, -148.0673218, 148.1857605
9: -61.4201469, 67.8731384, -61.6307106, 68.0492859, -129.4694366, 129.5038452
10: -91.2995071, 89.5029907, -91.6110382, 89.7907410, -181.0902100, 181.1140289
11: -83.1539993, 42.1780014, -83.5164795, 42.3634377, -125.5174408, 125.6944733
12: -60.1816673, 76.2200089, -60.3480492, 76.6107025, -136.7923584, 136.5680542
13: -67.4642792, 104.5762405, -67.8403931, 105.0800018, -172.5442810, 172.4166260
14: -116.5173035, 60.5877304, -116.8563309, 60.6746101, -177.1918945, 177.4440613
15: -66.6384659, 63.2988892, -66.8784485, 63.4283371, -130.0668030, 130.1773376
16: -97.1483307, 53.7839355, -97.4485245, 54.0349197, -151.1832581, 151.2324524
17: -109.2216034, 72.0584183, -109.6609344, 72.3331985, -181.5548096, 181.7193604
18: -90.7939453, 46.3641052, -91.1814270, 46.6382713, -137.4322205, 137.5455017
19: -66.8483353, 34.8391914, -67.0874481, 34.9974518, -101.8457794, 101.9266357
20: -65.4928970, 42.4775391, -65.7723999, 42.6343689, -108.1272659, 108.2499390
21: -84.6232452, 42.6506577, -84.9744720, 42.8346596, -127.4578857, 127.6251297
22: -74.7109070, 61.9340210, -75.1176758, 62.1032028, -136.8141174, 137.0516968
23: -66.7292862, 48.0296631, -66.9837875, 48.2200813, -114.9493713, 115.0134506
24: -85.4085846, 55.2915878, -85.7830505, 55.5042877, -140.9128723, 141.0746460
25: -62.8385201, 57.7036858, -63.1237679, 57.8964424, -120.7349625, 120.8274536
26: -90.3930969, 57.1230392, -90.8057404, 57.3738976, -147.7669678, 147.9287720
27: -98.7385178, 45.8036652, -99.2059402, 46.0489426, -144.7874603, 145.0096130
28: -66.4168701, 53.0838432, -66.6704788, 53.2728386, -119.6897125, 119.7543106
29: -77.7019806, 65.3025360, -78.1632462, 65.4629288, -143.1649017, 143.4657898
30: -82.8703766, 55.7238503, -83.2483521, 55.9116287, -138.7819977, 138.9721985
31: -89.4996414, 48.7630463, -89.8078461, 48.9821739, -138.4818115, 138.5708923
32: -76.6023102, 59.2982941, -76.8010635, 59.5152321, -136.1175385, 136.0993500
33: -113.0542755, 70.5636520, -113.2731476, 70.9152298, -183.9694824, 183.8367920
34: -90.0221329, 51.8283691, -90.1832962, 52.1422348, -142.1643677, 142.0116577
35: -88.5199585, 63.4256897, -88.6915436, 63.8034782, -152.3234406, 152.1172333
36: -87.6072083, 60.3084564, -87.7812653, 60.5777397, -148.1849518, 148.0897217
37: -135.9238586, 45.0197830, -136.1818237, 45.3048553, -181.2287140, 181.2015991
38: -109.3033371, 68.1405945, -109.4968872, 68.5823135, -177.8856201, 177.6374817
39: -121.5988235, 67.9123230, -121.9166870, 68.3329773, -189.9317932, 189.8290100
40: -112.5378799, 35.2819824, -112.7947693, 35.4760437, -148.0139160, 148.0767517
41: -85.9116974, 49.2652206, -86.0639191, 49.5146942, -135.4263916, 135.3291321
42: -58.1160355, 46.7487946, -58.2710571, 46.9624062, -105.0784454, 105.0198517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=604, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1569

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7984825, upper bound: 115.6829148
time: 134.88 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7984825, upper bound: 115.8148526
time: 136.48 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -103.3201981, 67.3104401, -103.4482193, 67.3569412, -170.6771240, 170.7586670
1: -49.8554611, 50.6036530, -49.9372559, 50.6635284, -100.5189819, 100.5408936
2: -48.4599571, 49.9681587, -48.5461731, 50.0388870, -98.4988403, 98.5143280
3: -52.0090218, 63.1224861, -52.1445618, 63.2185287, -115.2275467, 115.2670441
4: -65.4540405, 56.9002151, -65.5823822, 56.9531250, -122.4071655, 122.4825974
5: -55.1237068, 58.8798561, -55.2278709, 58.9490204, -114.0727234, 114.1077194
6: -86.3234406, 50.4500351, -86.3729095, 50.6117439, -136.9351654, 136.8229370
7: -67.2074280, 49.1043549, -67.3116150, 49.1526337, -116.3600616, 116.4159622
8: -78.5234680, 69.6688690, -78.6839294, 69.6863937, -148.2098694, 148.3527832
9: -61.5728607, 67.9731140, -61.6631470, 68.0648651, -129.6377258, 129.6362610
10: -91.5848846, 89.6235199, -91.6772156, 89.8087463, -181.3936157, 181.3007355
11: -83.5295486, 42.3358231, -83.6042786, 42.3792381, -125.9087830, 125.9401016
12: -60.4626198, 76.4931641, -60.3673439, 76.6780548, -137.1406708, 136.8605042
13: -67.7946548, 104.8633804, -67.8676529, 105.1493301, -172.9439850, 172.7310333
14: -116.7472229, 60.6509857, -116.8991318, 60.6842346, -177.4314575, 177.5501099
15: -66.7493134, 63.3892212, -66.9019318, 63.4378319, -130.1871338, 130.2911377
16: -97.4316711, 53.9133492, -97.5082550, 54.0522232, -151.4838867, 151.4216003
17: -109.5388107, 72.3352966, -109.7299042, 72.4001312, -181.9389343, 182.0652008
18: -90.9986191, 46.5001259, -91.2253494, 46.6612320, -137.6598206, 137.7254791
19: -67.0228424, 34.9138412, -67.1266785, 35.0073318, -102.0301743, 102.0405045
20: -65.7110291, 42.6781235, -65.8209381, 42.6476974, -108.3587189, 108.4990616
21: -84.9419937, 42.8699760, -85.0482712, 42.8471413, -127.7891235, 127.9182434
22: -74.9610443, 62.0685844, -75.1642456, 62.1159477, -137.0769806, 137.2328186
23: -66.8868408, 48.1531105, -67.0173798, 48.2333870, -115.1202240, 115.1704865
24: -85.6456375, 55.4884605, -85.8324661, 55.5187454, -141.1643677, 141.3209229
25: -63.0384712, 57.8585358, -63.1659470, 57.9090462, -120.9475174, 121.0244827
26: -90.5764923, 57.2772064, -90.8373566, 57.3945045, -147.9709930, 148.1145630
27: -99.0130234, 46.0607758, -99.2675323, 46.0666542, -145.0796814, 145.3283081
28: -66.5119629, 53.1837692, -66.6905060, 53.2883148, -119.8002777, 119.8742752
29: -78.0550613, 65.4719238, -78.2373810, 65.4746094, -143.5296631, 143.7092896
30: -83.1860352, 56.0167656, -83.3184967, 55.9283218, -139.1143494, 139.3352661
31: -89.7394409, 48.8876381, -89.8586273, 48.9955406, -138.7349854, 138.7462616
32: -76.7618103, 59.4349213, -76.8161011, 59.5451012, -136.3069153, 136.2510071
33: -113.2949753, 70.8429108, -113.2917175, 70.9865341, -184.2815094, 184.1346283
34: -90.1483917, 52.1861420, -90.1957321, 52.2347565, -142.3831482, 142.3818665
35: -88.7794037, 63.7875023, -88.7069016, 63.9018860, -152.6812897, 152.4943848
36: -87.8493576, 60.5940742, -87.7944870, 60.6525269, -148.5018921, 148.3885498
37: -136.2288208, 45.2773781, -136.2062836, 45.3706779, -181.5994873, 181.4836578
38: -109.5328522, 68.5898743, -109.5121918, 68.6992035, -178.2320557, 178.1020660
39: -121.8772354, 68.2198410, -121.9391708, 68.4108582, -190.2880859, 190.1590118
40: -112.6991959, 35.4329948, -112.8109665, 35.5117188, -148.2109070, 148.2439575
41: -86.0843887, 49.4597321, -86.0727692, 49.5597725, -135.6441650, 135.5325012
42: -58.2223053, 46.8264732, -58.2849960, 46.9779892, -105.2002869, 105.1114655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 576

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7984825, upper bound: 115.7878382
time: 178.76 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7984825, upper bound: 115.9196629
time: 89.55 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -103.3348770, 67.4136200, -103.3231964, 67.2818451, -170.6167297, 170.7368011
1: -49.8611031, 50.6482468, -49.8471680, 50.5670624, -100.4281616, 100.4954071
2: -48.4915810, 50.0620804, -48.4752121, 49.9509354, -98.4425201, 98.5372849
3: -52.0377617, 63.2369766, -52.0239563, 63.1099625, -115.1477203, 115.2609329
4: -65.5373917, 56.9788551, -65.5218506, 56.8700485, -122.4074402, 122.5007019
5: -55.0689583, 58.9778404, -55.1123886, 58.8394928, -113.9084473, 114.0902252
6: -86.3217010, 50.4806709, -86.2522507, 50.4477158, -136.7694092, 136.7329102
7: -67.1304321, 49.1664352, -67.1588898, 49.0413208, -116.1717529, 116.3253250
8: -78.5979462, 69.7348938, -78.6078033, 69.5961761, -148.1941223, 148.3426819
9: -61.5538330, 68.0186539, -61.5365486, 67.9408875, -129.4947205, 129.5552063
10: -91.4459686, 89.6127701, -91.4536819, 89.5922165, -181.0381775, 181.0664520
11: -83.4073715, 42.3256226, -83.3978271, 42.3180580, -125.7254333, 125.7234497
12: -60.2997742, 76.4072952, -60.2537727, 76.4504166, -136.7501831, 136.6610718
13: -67.7861557, 104.9697266, -67.6481628, 104.7981262, -172.5842896, 172.6178894
14: -116.7537384, 60.7039871, -116.7273254, 60.5667419, -177.3204803, 177.4313049
15: -66.7670364, 63.3819466, -66.7559357, 63.3153419, -130.0823822, 130.1378784
16: -97.2987366, 53.8903084, -97.3107681, 53.8515472, -151.1502686, 151.2010803
17: -109.5087128, 72.3451920, -109.5490952, 72.2324753, -181.7411804, 181.8942871
18: -91.0868378, 46.5761032, -90.9527740, 46.4271431, -137.5139465, 137.5288696
19: -67.0488434, 34.9734879, -66.9730453, 34.9413033, -101.9901428, 101.9465332
20: -65.6503448, 42.6071434, -65.6455231, 42.5490074, -108.1993561, 108.2526703
21: -84.8594208, 42.8168716, -84.8336105, 42.7680931, -127.6275177, 127.6504669
22: -74.9222488, 62.0625801, -74.8641739, 61.9501305, -136.8723755, 136.9267578
23: -66.9729691, 48.2101860, -66.8425751, 48.1554565, -115.1284256, 115.0527573
24: -85.6430817, 55.4666405, -85.5823059, 55.3918266, -141.0349121, 141.0489502
25: -63.0498924, 57.8834152, -62.9780388, 57.8135796, -120.8634720, 120.8614502
26: -90.6793442, 57.3139954, -90.5237732, 57.1687813, -147.8481293, 147.8377686
27: -99.0251312, 46.0054817, -98.9442749, 45.8876762, -144.9127808, 144.9497375
28: -66.6248245, 53.2488976, -66.4915161, 53.1658363, -119.7906494, 119.7404099
29: -77.9327621, 65.4404297, -77.9376907, 65.3479233, -143.2806702, 143.3781128
30: -83.0441895, 55.8893890, -83.0816193, 55.8201904, -138.8643799, 138.9710083
31: -89.7754059, 48.9581375, -89.6671982, 48.9204025, -138.6958008, 138.6253357
32: -76.7442398, 59.4293213, -76.6646118, 59.4117432, -136.1559753, 136.0939331
33: -113.2442017, 70.7858658, -113.1599808, 70.7818604, -184.0260620, 183.9458466
34: -90.1486053, 51.9233093, -90.0935059, 52.0670090, -142.2156067, 142.0168152
35: -88.6489105, 63.5737953, -88.5847473, 63.6991425, -152.3480530, 152.1585388
36: -87.7422104, 60.4006805, -87.6864243, 60.5253983, -148.2676086, 148.0870972
37: -136.1275940, 45.1334534, -136.0897217, 45.2236023, -181.3511963, 181.2231750
38: -109.5067673, 68.2631683, -109.3970184, 68.4545746, -177.9613342, 177.6601868
39: -121.8858871, 68.1571350, -121.7453079, 68.1346436, -190.0205078, 189.9024353
40: -112.7245636, 35.4164352, -112.6833038, 35.3809967, -148.1055603, 148.0997314
41: -86.0275574, 49.3353806, -85.9363403, 49.4161301, -135.4436798, 135.2717285
42: -58.1978607, 46.8394012, -58.1503105, 46.8301010, -105.0279617, 104.9897003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=604, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 576

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7979509, upper bound: 115.5525535
time: 754.69 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7979509, upper bound: 115.6858120
time: 181.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 938.51 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 938.51
Output dim: 13, lower bound: -115.7979509, upper bound: 115.5525535
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 938.51
Output dim: 13, lower bound: -115.7979509, upper bound: 115.6858120
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 938.51
Output dim: 13, lower bound: -115.7979509, upper bound: 115.6574372
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 938.51
Output dim: 13, lower bound: -115.7979509, upper bound: 115.7907265
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 938.51
Output dim: 13, lower bound: -115.7984825, upper bound: 115.6829148
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 938.51
Output dim: 13, lower bound: -115.7984825, upper bound: 115.8148526
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 938.51
Output dim: 13, lower bound: -115.7984825, upper bound: 115.7878382
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 938.51
Output dim: 13, lower bound: -115.7984825, upper bound: 115.9196629
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 938.51
Output dim: 13, lower bound: -115.7979509, upper bound: 115.5525535
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 938.51
Output dim: 13, lower bound: -115.7979509, upper bound: 115.6858120
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 938.51
Output dim: 13, lower bound: -115.8029113, upper bound: 115.7964858
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 938.51
Output dim: 13, lower bound: -115.9245932, upper bound: 115.8199144
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 938.51
Output dim: 13, lower bound: -115.8034167, upper bound: 115.9245929
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=172.8270263671875
rel_dist={13: [-115.95178022674816, 115.95178022664516]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1725

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9242644, upper bound: 114.0376414
time: 139.72 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0376412, upper bound: 114.0376414
time: 104.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 244.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 244.69
Output dim: 13, lower bound: -113.9242644, upper bound: 114.0376414
IS_A2, status: Status.UNKNOWN, split count: 1, time: 244.69
Output dim: 13, lower bound: -114.0376412, upper bound: 114.0376414

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -103.2789230, 67.2919617, -103.3918686, 67.3217163, -170.6006470, 170.6838379
1: -49.8466568, 50.5750504, -49.9175186, 50.5982361, -100.4448853, 100.4925690
2: -48.4513588, 49.9596710, -48.5266991, 49.9814682, -98.4328308, 98.4863739
3: -52.0105362, 63.1161232, -52.1075172, 63.1439514, -115.1544800, 115.2236404
4: -65.4317474, 56.8847351, -65.5458679, 56.9087067, -122.3404541, 122.4306030
5: -55.1246986, 58.8373146, -55.2030449, 58.8690720, -113.9937744, 114.0403595
6: -86.2812881, 50.4314461, -86.3340302, 50.4826736, -136.7639618, 136.7654724
7: -67.1984406, 49.0380058, -67.2889099, 49.0663033, -116.2647400, 116.3269196
8: -78.5159607, 69.6070328, -78.6356812, 69.6432800, -148.1592407, 148.2427063
9: -61.5627823, 67.9424744, -61.6321487, 67.9700775, -129.5328674, 129.5746155
10: -91.5939331, 89.5898285, -91.6506958, 89.6374893, -181.2314148, 181.2405090
11: -83.5207901, 42.2501907, -83.5541611, 42.3363495, -125.8571320, 125.8043518
12: -60.2727776, 76.4904327, -60.3341217, 76.5498657, -136.8226318, 136.8245544
13: -67.5933228, 104.8646698, -67.8187714, 104.8983078, -172.4916382, 172.6834412
14: -116.6980896, 60.6436386, -116.8016891, 60.6679230, -177.3660126, 177.4453278
15: -66.7402344, 63.3593292, -66.8037262, 63.4105339, -130.1507721, 130.1630554
16: -97.4204788, 53.8608360, -97.4757538, 53.8955612, -151.3160400, 151.3365936
17: -109.5036087, 72.3375397, -109.6444397, 72.3669739, -181.8705750, 181.9819641
18: -90.9898911, 46.4713249, -91.0276031, 46.6247787, -137.6146545, 137.4989319
19: -67.0177841, 34.8906326, -67.0460892, 34.9847412, -102.0025253, 101.9367142
20: -65.6934586, 42.5447807, -65.7213974, 42.6305733, -108.3240280, 108.2661667
21: -84.9193420, 42.7168999, -84.9523468, 42.8299904, -127.7493286, 127.6692429
22: -74.9042816, 62.0042725, -74.9447784, 62.0978889, -137.0021667, 136.9490509
23: -66.8766403, 48.0928650, -66.9011993, 48.2098503, -115.0864868, 114.9940567
24: -85.6265869, 55.3728523, -85.6597748, 55.4959641, -141.1225586, 141.0326233
25: -63.0124893, 57.7699051, -63.0438919, 57.8875961, -120.9000854, 120.8137970
26: -90.5337219, 57.2201385, -90.5833893, 57.3631973, -147.8969116, 147.8035278
27: -99.0048828, 45.8935585, -99.0389557, 46.0436630, -145.0485382, 144.9325104
28: -66.5087280, 53.1532402, -66.5349426, 53.2684517, -119.7771759, 119.6881790
29: -78.0079727, 65.3635406, -78.0453796, 65.4584427, -143.4664154, 143.4089203
30: -83.1591110, 55.8056870, -83.1892624, 55.9092293, -139.0683441, 138.9949341
31: -89.7233124, 48.8365593, -89.7612762, 48.9665260, -138.6898346, 138.5978394
32: -76.6843872, 59.4214249, -76.7618484, 59.4591064, -136.1434937, 136.1832733
33: -113.1405029, 70.8516159, -113.2460098, 70.8879166, -184.0284119, 184.0976257
34: -90.0883026, 52.1685448, -90.1539078, 52.1924553, -142.2807617, 142.3224487
35: -88.5957031, 63.7896461, -88.6690521, 63.8206940, -152.4163971, 152.4586945
36: -87.6762085, 60.5937653, -87.7514572, 60.6180267, -148.2942200, 148.3452148
37: -136.0542297, 45.2893219, -136.1416321, 45.3213425, -181.3755493, 181.4309540
38: -109.3837280, 68.5697327, -109.4704590, 68.6015778, -177.9853058, 178.0401917
39: -121.7112503, 68.2278595, -121.8789368, 68.2468872, -189.9581299, 190.1067963
40: -112.6575012, 35.4271545, -112.7358551, 35.4528542, -148.1103516, 148.1630096
41: -85.9746704, 49.4464340, -86.0300522, 49.4823456, -135.4570007, 135.4764862
42: -58.2059669, 46.8197632, -58.2460518, 46.8665962, -105.0725632, 105.0658112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=461, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1592

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.9170512, upper bound: 113.9191859
time: 145.21 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9170512, upper bound: 114.0303959
time: 180.52 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -103.4987564, 67.4807739, -103.4313812, 67.3317566, -170.8305054, 170.9121552
1: -49.9841461, 50.7235031, -49.9422798, 50.6057663, -100.5899124, 100.6657867
2: -48.5848351, 50.1187897, -48.5566597, 49.9886627, -98.5734863, 98.6754456
3: -52.1724319, 63.3241463, -52.1426659, 63.1532974, -115.3257294, 115.4668121
4: -65.6181717, 57.0589294, -65.5835876, 56.9163208, -122.5344925, 122.6425095
5: -55.2560272, 59.0489693, -55.2294807, 58.8796234, -114.1356506, 114.2784500
6: -86.4045105, 50.5677490, -86.3511047, 50.5008125, -136.9053192, 136.9188538
7: -67.3689041, 49.2245445, -67.3197708, 49.0759277, -116.4448318, 116.5443115
8: -78.7240067, 69.8168335, -78.6781158, 69.6556320, -148.3796387, 148.4949493
9: -61.7092667, 68.0905609, -61.6542015, 67.9785538, -129.6878052, 129.7447510
10: -91.7617569, 89.7050629, -91.6705322, 89.6526031, -181.4143524, 181.3755951
11: -83.7840271, 42.4016991, -83.5641556, 42.3674469, -126.1514587, 125.9658508
12: -60.4028091, 76.6851578, -60.3545837, 76.5699387, -136.9727478, 137.0397339
13: -67.9509506, 105.2626648, -67.9014053, 104.9089203, -172.8598633, 173.1640625
14: -116.9430313, 60.7728233, -116.8363113, 60.6756821, -177.6187134, 177.6091309
15: -66.8750763, 63.4576836, -66.8258286, 63.4256058, -130.3006744, 130.2835083
16: -97.5842438, 53.9723434, -97.4944229, 53.9066505, -151.4908905, 151.4667664
17: -109.7997818, 72.6318970, -109.6938324, 72.3767395, -182.1764832, 182.3257294
18: -91.2873077, 46.7206841, -91.0398254, 46.6805840, -137.9678955, 137.7605133
19: -67.2232437, 35.0353203, -67.0546188, 35.0189209, -102.2421646, 102.0899277
20: -65.8555374, 42.6894531, -65.7295227, 42.6612015, -108.5167313, 108.4189758
21: -85.1621933, 42.8966904, -84.9614563, 42.8708534, -128.0330353, 127.8581467
22: -75.1213989, 62.1596489, -74.9575806, 62.1314926, -137.2528992, 137.1172180
23: -67.1238098, 48.2861977, -66.9077301, 48.2524033, -115.3762131, 115.1939240
24: -85.8645935, 55.5689545, -85.6694794, 55.5410080, -141.4055939, 141.2384338
25: -63.2279701, 57.9655228, -63.0534248, 57.9300385, -121.1580048, 121.0189514
26: -90.8253021, 57.4471397, -90.5992279, 57.4147034, -148.2399902, 148.0463715
27: -99.2961731, 46.1256485, -99.0487061, 46.0986061, -145.3947754, 145.1743469
28: -66.7207642, 53.3386803, -66.5425415, 53.3101730, -120.0309296, 119.8812180
29: -78.2435379, 65.5223541, -78.0568085, 65.4940643, -143.7376099, 143.5791626
30: -83.3387451, 55.9877357, -83.1979980, 55.9465675, -139.2852936, 139.1857300
31: -90.0051270, 49.0414886, -89.7728729, 49.0136070, -139.0187225, 138.8143616
32: -76.8434982, 59.5563354, -76.7878265, 59.4723167, -136.3158112, 136.3441620
33: -113.3452225, 71.0796661, -113.2819672, 70.8999634, -184.2451782, 184.3616333
34: -90.2239685, 52.2682648, -90.1740570, 52.1996765, -142.4236450, 142.4423218
35: -88.7385101, 63.9412918, -88.6912537, 63.8313484, -152.5698547, 152.6325378
36: -87.8224945, 60.6886368, -87.7748489, 60.6260567, -148.4485474, 148.4634705
37: -136.2642822, 45.4085770, -136.1705017, 45.3321266, -181.5964050, 181.5790405
38: -109.5997543, 68.6971359, -109.5005951, 68.6123810, -178.2121277, 178.1977234
39: -122.0251160, 68.4777374, -121.9418945, 68.2528000, -190.2779083, 190.4196320
40: -112.8514175, 35.5672913, -112.7621384, 35.4607201, -148.3121338, 148.3294373
41: -86.1057053, 49.5208855, -86.0487366, 49.4935837, -135.5992889, 135.5696259
42: -58.3013573, 46.9149704, -58.2594719, 46.8812828, -105.1826401, 105.1744385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=461, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1592

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.9170512, upper bound: 113.9191859
time: 110.54 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9170512, upper bound: 114.0303959
time: 104.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 217.91 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 217.91
Output dim: 13, lower bound: -113.9170512, upper bound: 113.9191859
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 217.91
Output dim: 13, lower bound: -113.9170512, upper bound: 114.0303959
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 217.91
Output dim: 13, lower bound: -113.9170512, upper bound: 113.9191859
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 217.91
Output dim: 13, lower bound: -113.9170512, upper bound: 114.0303959

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -103.2711487, 67.2897644, -103.4435120, 67.3598175, -170.6309662, 170.7332764
1: -49.8428154, 50.5736351, -49.9330559, 50.6664009, -100.5092163, 100.5066833
2: -48.4475441, 49.9582596, -48.5428734, 50.0414352, -98.4889679, 98.5011292
3: -52.0048103, 63.1140289, -52.1395950, 63.2313919, -115.2361984, 115.2536240
4: -65.4293823, 56.8829536, -65.5738220, 56.9583626, -122.3877258, 122.4567719
5: -55.1195297, 58.8356590, -55.2241631, 58.9513206, -114.0708237, 114.0598221
6: -86.2771912, 50.4293518, -86.3748322, 50.6104164, -136.8875885, 136.8041840
7: -67.1923523, 49.0366364, -67.3066177, 49.1529922, -116.3453445, 116.3432541
8: -78.5139771, 69.6040955, -78.6768036, 69.6880798, -148.2020569, 148.2808990
9: -61.5591545, 67.9407806, -61.6644936, 68.0654449, -129.6245880, 129.6052704
10: -91.5875702, 89.5873642, -91.6874161, 89.8093414, -181.3969116, 181.2747803
11: -83.5162125, 42.2480011, -83.6118011, 42.3748016, -125.8910141, 125.8598022
12: -60.2688255, 76.4880676, -60.3655891, 76.6780243, -136.9468536, 136.8536530
13: -67.5839310, 104.8619690, -67.8498077, 105.1539612, -172.7378845, 172.7117767
14: -116.6941452, 60.6365891, -116.8951569, 60.6866417, -177.3807831, 177.5317383
15: -66.7375488, 63.3548546, -66.8997574, 63.4384880, -130.1760406, 130.2546082
16: -97.4101562, 53.8585815, -97.5164642, 54.0519104, -151.4620667, 151.3750458
17: -109.4982758, 72.3309326, -109.7230072, 72.4090500, -181.9073181, 182.0539398
18: -90.9868011, 46.4610443, -91.2301483, 46.6514435, -137.6382446, 137.6911926
19: -67.0154419, 34.8876991, -67.1322784, 35.0006218, -102.0160675, 102.0199738
20: -65.6914825, 42.5404510, -65.8228073, 42.6430206, -108.3345032, 108.3632584
21: -84.9162216, 42.7127838, -85.0534592, 42.8400269, -127.7562485, 127.7662430
22: -74.9013367, 61.9975777, -75.1662598, 62.1108093, -137.0121460, 137.1638184
23: -66.8744659, 48.0890617, -67.0207520, 48.2242889, -115.0987549, 115.1097946
24: -85.6236725, 55.3672829, -85.8376923, 55.5118828, -141.1355591, 141.2049713
25: -63.0105782, 57.7649078, -63.1680145, 57.9007683, -120.9113464, 120.9329147
26: -90.5308228, 57.2109032, -90.8382263, 57.3842888, -147.9150696, 148.0491180
27: -99.0023193, 45.8853874, -99.2734222, 46.0559273, -145.0582275, 145.1588135
28: -66.5070419, 53.1476059, -66.6922073, 53.2787056, -119.7857513, 119.8398132
29: -78.0046844, 65.3576050, -78.2420197, 65.4677429, -143.4724274, 143.5996094
30: -83.1564331, 55.8002892, -83.3222733, 55.9230042, -139.0794220, 139.1225433
31: -89.7201996, 48.8300667, -89.8668823, 48.9866066, -138.7067871, 138.6969452
32: -76.6801605, 59.4195290, -76.8144531, 59.5451965, -136.2253571, 136.2339783
33: -113.1347351, 70.8490067, -113.2849579, 70.9898300, -184.1245422, 184.1339722
34: -90.0839844, 52.1662445, -90.1938934, 52.2394562, -142.3234406, 142.3601379
35: -88.5912704, 63.7873573, -88.7038727, 63.9053497, -152.4966125, 152.4912262
36: -87.6726913, 60.5925446, -87.7915344, 60.6579590, -148.3306580, 148.3840790
37: -136.0506897, 45.2871857, -136.2080536, 45.3734589, -181.4241486, 181.4952393
38: -109.3790054, 68.5673981, -109.5092316, 68.7041473, -178.0831604, 178.0766296
39: -121.7040176, 68.2256927, -121.9287415, 68.4155884, -190.1195984, 190.1544342
40: -112.6550140, 35.4239731, -112.8192368, 35.5124092, -148.1674194, 148.2432098
41: -85.9698868, 49.4447403, -86.0743408, 49.5601425, -135.5300293, 135.5190735
42: -58.2018242, 46.8174553, -58.2908325, 46.9767342, -105.1785583, 105.1082764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 693

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.9146113, upper bound: 113.9320451
time: 253.55 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9146113, upper bound: 114.0279602
time: 129.28 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -103.4909821, 67.4785843, -103.4830475, 67.3698807, -170.8608704, 170.9616394
1: -49.9803009, 50.7221069, -49.9577866, 50.6739464, -100.6542511, 100.6798859
2: -48.5810280, 50.1173897, -48.5727501, 50.0486221, -98.6296539, 98.6901398
3: -52.1666832, 63.3220291, -52.1747589, 63.2407074, -115.4073944, 115.4967880
4: -65.6157990, 57.0571518, -65.6115112, 56.9659119, -122.5817032, 122.6686630
5: -55.2508392, 59.0473289, -55.2506065, 58.9618416, -114.2126694, 114.2979355
6: -86.4004211, 50.5656128, -86.3919296, 50.6285858, -137.0289917, 136.9575500
7: -67.3628311, 49.2231827, -67.3374405, 49.1625900, -116.5254211, 116.5606232
8: -78.7219925, 69.8139038, -78.7192383, 69.7005005, -148.4224854, 148.5331421
9: -61.7056198, 68.0888824, -61.6865501, 68.0739365, -129.7795410, 129.7754211
10: -91.7553406, 89.7026443, -91.7072601, 89.8245087, -181.5798492, 181.4098969
11: -83.7794571, 42.3994980, -83.6217880, 42.4058838, -126.1853409, 126.0212860
12: -60.3988762, 76.6828613, -60.3860741, 76.6980972, -137.0969696, 137.0689240
13: -67.9415817, 105.2600098, -67.9324036, 105.1645813, -173.1061707, 173.1924133
14: -116.9391251, 60.7657852, -116.9296265, 60.6944656, -177.6335754, 177.6954041
15: -66.8724060, 63.4532204, -66.9218445, 63.4535828, -130.3259888, 130.3750610
16: -97.5739059, 53.9700699, -97.5351410, 54.0629807, -151.6368866, 151.5052185
17: -109.7944565, 72.6253357, -109.7723312, 72.4188538, -182.2133179, 182.3976593
18: -91.2841949, 46.7104111, -91.2423401, 46.7072182, -137.9913940, 137.9527435
19: -67.2209091, 35.0323334, -67.1407700, 35.0348053, -102.2557068, 102.1730957
20: -65.8535690, 42.6851501, -65.8309402, 42.6736107, -108.5271759, 108.5160828
21: -85.1590958, 42.8925514, -85.0625687, 42.8809204, -128.0400085, 127.9551163
22: -75.1184540, 62.1529465, -75.1790771, 62.1444092, -137.2628632, 137.3320312
23: -67.1216278, 48.2823944, -67.0272446, 48.2668266, -115.3884583, 115.3096390
24: -85.8616409, 55.5634460, -85.8473892, 55.5569534, -141.4185791, 141.4108276
25: -63.2260399, 57.9604950, -63.1775246, 57.9432373, -121.1692734, 121.1380157
26: -90.8224030, 57.4379158, -90.8540573, 57.4358063, -148.2581940, 148.2919617
27: -99.2936325, 46.1174278, -99.2831726, 46.1108398, -145.4044800, 145.4006042
28: -66.7191010, 53.3330536, -66.6998215, 53.3204041, -120.0395050, 120.0328751
29: -78.2402878, 65.5164642, -78.2534409, 65.5033188, -143.7435913, 143.7698975
30: -83.3360748, 55.9823532, -83.3309784, 55.9603615, -139.2964325, 139.3133240
31: -90.0019989, 49.0349884, -89.8784943, 49.0336838, -139.0356750, 138.9134827
32: -76.8392639, 59.5544891, -76.8404007, 59.5583954, -136.3976593, 136.3948975
33: -113.3394623, 71.0770950, -113.3208542, 71.0018921, -184.3413544, 184.3979340
34: -90.2195892, 52.2660027, -90.2140732, 52.2466812, -142.4662476, 142.4800720
35: -88.7340698, 63.9389801, -88.7260895, 63.9160194, -152.6500854, 152.6650696
36: -87.8189621, 60.6873894, -87.8149414, 60.6659889, -148.4849396, 148.5023193
37: -136.2607422, 45.4064941, -136.2369232, 45.3841858, -181.6449280, 181.6434174
38: -109.5950089, 68.6948395, -109.5394135, 68.7150116, -178.3100281, 178.2342529
39: -122.0178375, 68.4755402, -121.9916382, 68.4215546, -190.4393921, 190.4671783
40: -112.8489456, 35.5641251, -112.8454742, 35.5202484, -148.3692017, 148.4096069
41: -86.1009216, 49.5192261, -86.0930023, 49.5713577, -135.6722717, 135.6122284
42: -58.2972260, 46.9126740, -58.3042488, 46.9914169, -105.2886353, 105.2169113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=460, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 693

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.9146113, upper bound: 113.9320451
time: 112.56 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9146113, upper bound: 114.0279602
time: 321.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 436.89 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 436.89
Output dim: 13, lower bound: -113.9146113, upper bound: 113.9320451
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 436.89
Output dim: 13, lower bound: -113.9146113, upper bound: 114.0279602
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 436.89
Output dim: 13, lower bound: -113.9146113, upper bound: 113.9320451
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 436.89
Output dim: 13, lower bound: -113.9146113, upper bound: 114.0279602

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -103.3188553, 67.3100739, -103.4331894, 67.3520813, -170.6709290, 170.7432556
1: -49.8548088, 50.6034050, -49.9281845, 50.6595192, -100.5143280, 100.5315857
2: -48.4592819, 49.9679337, -48.5359192, 50.0351486, -98.4944305, 98.5038528
3: -52.0080490, 63.1221352, -52.1322174, 63.2111435, -115.2191925, 115.2543488
4: -65.4536438, 56.8999634, -65.5686417, 56.9483223, -122.4019470, 122.4685974
5: -55.1228065, 58.8795547, -55.2178307, 58.9439278, -114.0667343, 114.0973816
6: -86.3227386, 50.4497032, -86.3648376, 50.6049347, -136.9276733, 136.8145294
7: -67.2064819, 49.1041298, -67.3002167, 49.1485138, -116.3549805, 116.4043427
8: -78.5231323, 69.6683655, -78.6690369, 69.6808014, -148.2039337, 148.3373718
9: -61.5722427, 67.9728165, -61.6530418, 68.0607758, -129.6329956, 129.6258545
10: -91.5837936, 89.6231003, -91.6665955, 89.8017883, -181.3855591, 181.2897034
11: -83.5288086, 42.3354568, -83.5975494, 42.3683319, -125.8971252, 125.9330063
12: -60.4619446, 76.4927673, -60.3590851, 76.6696930, -137.1316376, 136.8518372
13: -67.7930450, 104.8629150, -67.8406677, 105.1433868, -172.9364319, 172.7035828
14: -116.7465439, 60.6498032, -116.8853149, 60.6802216, -177.4267578, 177.5350952
15: -66.7488556, 63.3884697, -66.8936157, 63.4302139, -130.1790771, 130.2820892
16: -97.4299774, 53.9129333, -97.4982452, 54.0474052, -151.4773712, 151.4111786
17: -109.5379562, 72.3341827, -109.7118530, 72.3937988, -181.9317627, 182.0460358
18: -90.9980698, 46.4983330, -91.2187958, 46.6422195, -137.6402893, 137.7171326
19: -67.0224457, 34.9133148, -67.1211853, 34.9959641, -102.0184097, 102.0345001
20: -65.7106781, 42.6773682, -65.8165207, 42.6368408, -108.3475113, 108.4938889
21: -84.9414825, 42.8692932, -85.0421982, 42.8332176, -127.7747040, 127.9114914
22: -74.9605255, 62.0674171, -75.1579742, 62.1041374, -137.0646667, 137.2253876
23: -66.8864746, 48.1524544, -67.0130310, 48.2194252, -115.1058960, 115.1654739
24: -85.6451035, 55.4875107, -85.8263474, 55.5031929, -141.1483002, 141.3138580
25: -63.0381432, 57.8576851, -63.1609383, 57.8947411, -120.9328766, 121.0186234
26: -90.5759888, 57.2756271, -90.8300552, 57.3770676, -147.9530334, 148.1056824
27: -99.0125809, 46.0594177, -99.2612991, 46.0484314, -145.0610046, 145.3207092
28: -66.5116959, 53.1828156, -66.6863556, 53.2746964, -119.7863770, 119.8691711
29: -78.0544968, 65.4708786, -78.2308350, 65.4626160, -143.5170898, 143.7017212
30: -83.1855774, 56.0158386, -83.3132172, 55.9151421, -139.1007080, 139.3290405
31: -89.7388916, 48.8864975, -89.8513565, 48.9796524, -138.7185364, 138.7378540
32: -76.7610931, 59.4345970, -76.8055725, 59.5397072, -136.3007965, 136.2401733
33: -113.2939987, 70.8424530, -113.2784576, 70.9805527, -184.2745514, 184.1209106
34: -90.1476212, 52.1857719, -90.1869659, 52.2303200, -142.3779449, 142.3727417
35: -88.7786331, 63.7871017, -88.6972733, 63.8966789, -152.6753082, 152.4843597
36: -87.8487549, 60.5938873, -87.7845535, 60.6481934, -148.4969482, 148.3784485
37: -136.2282104, 45.2770233, -136.1933594, 45.3654480, -181.5936432, 181.4703827
38: -109.5320053, 68.5894470, -109.5006561, 68.6934738, -178.2254791, 178.0900879
39: -121.8759842, 68.2194901, -121.9176102, 68.4069290, -190.2828979, 190.1371002
40: -112.6987610, 35.4324570, -112.7976990, 35.5079384, -148.2066956, 148.2301483
41: -86.0835876, 49.4594345, -86.0644379, 49.5546188, -135.6381989, 135.5238647
42: -58.2215919, 46.8260918, -58.2777824, 46.9716797, -105.1932678, 105.1038666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.9097408, upper bound: 113.9091641
time: 94.32 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9097408, upper bound: 114.0231334
time: 117.20 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -103.5387955, 67.4988556, -103.4726791, 67.3621292, -170.9009247, 170.9715271
1: -49.9924278, 50.7518463, -49.9529152, 50.6670570, -100.6594543, 100.7047577
2: -48.5928421, 50.1270218, -48.5658188, 50.0423317, -98.6351776, 98.6928329
3: -52.1699791, 63.3300667, -52.1673660, 63.2204781, -115.3904572, 115.4974365
4: -65.6401138, 57.0741234, -65.6063538, 56.9558907, -122.5960083, 122.6804657
5: -55.2541389, 59.0911255, -55.2442780, 58.9544716, -114.2086105, 114.3353882
6: -86.4459839, 50.5859451, -86.3819199, 50.6230774, -137.0690613, 136.9678650
7: -67.3770599, 49.2906418, -67.3310623, 49.1581039, -116.5351639, 116.6216965
8: -78.7312164, 69.8780746, -78.7114410, 69.6931915, -148.4243927, 148.5895081
9: -61.7187347, 68.1208725, -61.6750717, 68.0692444, -129.7879791, 129.7959290
10: -91.7515717, 89.7383881, -91.6864166, 89.8169479, -181.5684967, 181.4247894
11: -83.7920151, 42.4870377, -83.6075287, 42.3994675, -126.1914673, 126.0945663
12: -60.5919762, 76.6874771, -60.3795815, 76.6898270, -137.2817993, 137.0670624
13: -68.1506805, 105.2609100, -67.9232483, 105.1540375, -173.3047180, 173.1841583
14: -116.9918365, 60.7790070, -116.9198685, 60.6879845, -177.6798096, 177.6988678
15: -66.8836517, 63.4868469, -66.9156952, 63.4452972, -130.3289490, 130.4025421
16: -97.5937347, 54.0243073, -97.5169144, 54.0584488, -151.6521759, 151.5411987
17: -109.8343735, 72.6284943, -109.7612381, 72.4035645, -182.2379150, 182.3897400
18: -91.2954254, 46.7477455, -91.2309723, 46.6980476, -137.9934692, 137.9787140
19: -67.2278748, 35.0579910, -67.1296844, 35.0301399, -102.2580032, 102.1876755
20: -65.8727264, 42.8220787, -65.8246460, 42.6674347, -108.5401611, 108.6467209
21: -85.1843338, 43.0491447, -85.0513153, 42.8740692, -128.0584106, 128.1004639
22: -75.1776199, 62.2228355, -75.1707916, 62.1377220, -137.3153381, 137.3936310
23: -67.1336060, 48.3457947, -67.0195618, 48.2619743, -115.3955765, 115.3653488
24: -85.8830338, 55.6836739, -85.8360596, 55.5482292, -141.4312592, 141.5197144
25: -63.2536011, 58.0532799, -63.1704826, 57.9372063, -121.1908112, 121.2237549
26: -90.8674927, 57.5026169, -90.8459015, 57.4285889, -148.2960815, 148.3485107
27: -99.3039246, 46.2915726, -99.2710648, 46.1033669, -145.4072876, 145.5626373
28: -66.7237244, 53.3682785, -66.6939545, 53.3164291, -120.0401535, 120.0622330
29: -78.2900391, 65.6297684, -78.2422333, 65.4981766, -143.7882080, 143.8719940
30: -83.3651810, 56.1978531, -83.3219223, 55.9524727, -139.3176575, 139.5197754
31: -90.0206528, 49.0914764, -89.8629456, 49.0267296, -139.0473785, 138.9544220
32: -76.9201965, 59.5694923, -76.8315201, 59.5529175, -136.4731140, 136.4010010
33: -113.4989014, 71.0704651, -113.3143539, 70.9926147, -184.4915009, 184.3848267
34: -90.2833252, 52.2854843, -90.2071381, 52.2375107, -142.5208435, 142.4926147
35: -88.9214935, 63.9387856, -88.7194672, 63.9072838, -152.8287659, 152.6582336
36: -87.9950562, 60.6887131, -87.8079300, 60.6562386, -148.6512756, 148.4966431
37: -136.4381866, 45.3963356, -136.2221985, 45.3762283, -181.8144226, 181.6185303
38: -109.7480011, 68.7167664, -109.5308380, 68.7042007, -178.4522095, 178.2476044
39: -122.1899796, 68.4693298, -121.9805450, 68.4128876, -190.6028595, 190.4498749
40: -112.8927307, 35.5725708, -112.8239517, 35.5158310, -148.4085693, 148.3965149
41: -86.2146149, 49.5339470, -86.0831299, 49.5659103, -135.7805176, 135.6170654
42: -58.3170128, 46.9213257, -58.2911911, 46.9863548, -105.3033676, 105.2125092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=460, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1569

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.9097408, upper bound: 113.9091641
time: 112.35 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0231339, upper bound: 114.0231334
time: 95.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 210.58 seconds
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 210.58
Output dim: 13, lower bound: -113.9097408, upper bound: 113.9091641
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 210.58
Output dim: 13, lower bound: -113.9097408, upper bound: 114.0231334
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 210.58
Output dim: 13, lower bound: -113.9097408, upper bound: 113.9091641
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 210.58
Output dim: 13, lower bound: -114.0231339, upper bound: 114.0231334

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -103.3117981, 67.3025208, -103.5175781, 67.3895187, -170.7013245, 170.8200989
1: -49.8487663, 50.6007385, -49.9489212, 50.7058487, -100.5546036, 100.5496597
2: -48.4558372, 49.9655266, -48.5722351, 50.0809135, -98.5367508, 98.5377655
3: -51.9994583, 63.1186676, -52.1859703, 63.3163261, -115.3157806, 115.3046417
4: -65.4503021, 56.8879204, -65.6636353, 56.9842072, -122.4345093, 122.5515594
5: -55.1172295, 58.8767090, -55.2619858, 59.0373001, -114.1545258, 114.1386948
6: -86.3151321, 50.4466133, -86.4159317, 50.8061142, -137.1212463, 136.8625488
7: -67.1997147, 49.1016159, -67.3348160, 49.2357025, -116.4354172, 116.4364319
8: -78.5208359, 69.6636658, -78.7473145, 69.7192001, -148.2400360, 148.4109650
9: -61.5676346, 67.9694672, -61.6941528, 68.1543579, -129.7219849, 129.6636200
10: -91.5759201, 89.6184158, -91.7273331, 90.0193024, -181.5952148, 181.3457336
11: -83.5171127, 42.3321342, -83.6590195, 42.4707413, -125.9878540, 125.9911499
12: -60.4547424, 76.4893188, -60.3916969, 76.8751602, -137.3298950, 136.8810120
13: -67.7813416, 104.8593216, -67.8804932, 105.4153595, -173.1966858, 172.7398071
14: -116.7403488, 60.6379089, -117.0159454, 60.7073669, -177.4477234, 177.6538544
15: -66.7454529, 63.3820496, -67.0531311, 63.4664879, -130.2119446, 130.4351807
16: -97.4139786, 53.9092827, -97.5481339, 54.2519989, -151.6659851, 151.4574127
17: -109.5303955, 72.3242798, -109.8191833, 72.4521255, -181.9825134, 182.1434631
18: -90.9940186, 46.4882164, -91.4774170, 46.6897049, -137.6837006, 137.9656372
19: -67.0189819, 34.9093819, -67.2363892, 35.0185165, -102.0374908, 102.1457672
20: -65.7075806, 42.6717224, -65.9403381, 42.6615829, -108.3691483, 108.6120453
21: -84.9368057, 42.8646736, -85.1571503, 42.8620682, -127.7988739, 128.0218201
22: -74.9558105, 62.0593987, -75.4199905, 62.1296425, -137.0854492, 137.4793854
23: -66.8831787, 48.1473694, -67.1516953, 48.2434692, -115.1266403, 115.2990570
24: -85.6404572, 55.4811554, -86.0350113, 55.5286713, -141.1691284, 141.5161743
25: -63.0352211, 57.8512802, -63.2966003, 57.9158897, -120.9511108, 121.1478729
26: -90.5719147, 57.2637749, -91.1290131, 57.4209290, -147.9928436, 148.3927765
27: -99.0084610, 46.0502319, -99.5301056, 46.0751305, -145.0835876, 145.5803223
28: -66.5087585, 53.1758499, -66.8626556, 53.2923965, -119.8011475, 120.0384979
29: -78.0487671, 65.4643021, -78.4382401, 65.4813919, -143.5301514, 143.9025421
30: -83.1810455, 56.0104141, -83.4374542, 55.9592018, -139.1402435, 139.4478760
31: -89.7347641, 48.8809891, -89.9889603, 49.0072365, -138.7420044, 138.8699493
32: -76.7501221, 59.4325294, -76.8592072, 59.6526337, -136.4027557, 136.2917175
33: -113.2847748, 70.8387909, -113.3250275, 71.1224670, -184.4072266, 184.1638184
34: -90.1391449, 52.1824951, -90.2349701, 52.3373985, -142.4765320, 142.4174652
35: -88.7701721, 63.7844772, -88.7358704, 64.0185547, -152.7887115, 152.5203552
36: -87.8423004, 60.5924530, -87.8257980, 60.7016640, -148.5439606, 148.4182434
37: -136.2216797, 45.2735519, -136.2732544, 45.4726028, -181.6942749, 181.5468140
38: -109.5250473, 68.5861359, -109.5492859, 68.8020248, -178.3270721, 178.1354218
39: -121.8664627, 68.2163544, -121.9815903, 68.5929871, -190.4594421, 190.1979370
40: -112.6942291, 35.4284782, -112.8828506, 35.6084633, -148.3026886, 148.3113251
41: -86.0759277, 49.4571495, -86.1197739, 49.6707535, -135.7466736, 135.5769196
42: -58.2148743, 46.8223724, -58.3346672, 47.1592560, -105.3741302, 105.1570282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=459, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 692

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.9032252, upper bound: 113.9049453
time: 100.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9032252, upper bound: 114.0166456
time: 143.20 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -103.5317154, 67.4913025, -103.5570450, 67.3995972, -170.9313049, 171.0483398
1: -49.9863625, 50.7491608, -49.9736595, 50.7133675, -100.6997223, 100.7228165
2: -48.5893898, 50.1246109, -48.6021042, 50.0881004, -98.6774750, 98.7267075
3: -52.1613770, 63.3265915, -52.2210846, 63.3256569, -115.4870224, 115.5476685
4: -65.6367798, 57.0620880, -65.7012939, 56.9918404, -122.6286011, 122.7633820
5: -55.2485352, 59.0882568, -55.2884331, 59.0478325, -114.2963638, 114.3766937
6: -86.4383698, 50.5828819, -86.4330292, 50.8242493, -137.2626190, 137.0158997
7: -67.3702621, 49.2881012, -67.3656616, 49.2453308, -116.6155853, 116.6537628
8: -78.7289276, 69.8733673, -78.7897339, 69.7315674, -148.4604950, 148.6631012
9: -61.7141190, 68.1175079, -61.7162323, 68.1628418, -129.8769531, 129.8337402
10: -91.7437134, 89.7337036, -91.7471924, 90.0344162, -181.7781067, 181.4808960
11: -83.7803574, 42.4837608, -83.6690598, 42.5018082, -126.2821655, 126.1528168
12: -60.5847816, 76.6840057, -60.4121857, 76.8952713, -137.4800568, 137.0961761
13: -68.1390228, 105.2572784, -67.9631042, 105.4259720, -173.5650024, 173.2203827
14: -116.9855652, 60.7671051, -117.0502930, 60.7151642, -177.7007141, 177.8173828
15: -66.8802567, 63.4804306, -67.0752106, 63.4815445, -130.3618011, 130.5556335
16: -97.5776978, 54.0206604, -97.5668030, 54.2630386, -151.8407288, 151.5874634
17: -109.8268051, 72.6186676, -109.8684845, 72.4619293, -182.2887268, 182.4871521
18: -91.2913666, 46.7376251, -91.4895935, 46.7455177, -138.0368805, 138.2272186
19: -67.2244415, 35.0540466, -67.2449036, 35.0527115, -102.2771454, 102.2989426
20: -65.8696289, 42.8164062, -65.9484558, 42.6921806, -108.5618134, 108.7648621
21: -85.1796570, 43.0444336, -85.1662750, 42.9029846, -128.0826263, 128.2106934
22: -75.1729126, 62.2148018, -75.4327698, 62.1632652, -137.3361816, 137.6475525
23: -67.1303101, 48.3407135, -67.1582184, 48.2860413, -115.4163513, 115.4989319
24: -85.8784180, 55.6772957, -86.0447235, 55.5737114, -141.4521332, 141.7220154
25: -63.2506523, 58.0468941, -63.3061180, 57.9583702, -121.2090149, 121.3530045
26: -90.8634338, 57.4908104, -91.1448364, 57.4724464, -148.3358765, 148.6356506
27: -99.2997589, 46.2823334, -99.5398560, 46.1300354, -145.4297943, 145.8221893
28: -66.7207870, 53.3613319, -66.8702393, 53.3340988, -120.0548859, 120.2315598
29: -78.2842789, 65.6231537, -78.4496613, 65.5169525, -143.8012085, 144.0728149
30: -83.3606567, 56.1923904, -83.4461365, 55.9965363, -139.3571930, 139.6385193
31: -90.0165405, 49.0859413, -90.0005493, 49.0543480, -139.0708771, 139.0864868
32: -76.9092407, 59.5674362, -76.8851318, 59.6658478, -136.5750885, 136.4525757
33: -113.4896774, 71.0668488, -113.3608856, 71.1345367, -184.6242065, 184.4277191
34: -90.2748032, 52.2822800, -90.2551727, 52.3446579, -142.6194458, 142.5374451
35: -88.9130554, 63.9361076, -88.7580414, 64.0292206, -152.9422760, 152.6941528
36: -87.9885712, 60.6873245, -87.8492203, 60.7097282, -148.6983032, 148.5365448
37: -136.4316406, 45.3928604, -136.3021240, 45.4833641, -181.9149933, 181.6949768
38: -109.7410278, 68.7134552, -109.5794983, 68.8128204, -178.5538483, 178.2929535
39: -122.1804352, 68.4661636, -122.0445099, 68.5989227, -190.7793579, 190.5106812
40: -112.8881683, 35.5686493, -112.9090729, 35.6163712, -148.5045471, 148.4777222
41: -86.2069550, 49.5316467, -86.1384277, 49.6820107, -135.8889465, 135.6700745
42: -58.3102608, 46.9175797, -58.3480835, 47.1739311, -105.4841919, 105.2656631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=459, inp2_unstable=459, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 576

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 692

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.9032252, upper bound: 113.9049453
time: 115.96 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9032252, upper bound: 114.0166456
time: 95.18 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 213.60 seconds
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 213.60
Output dim: 13, lower bound: -113.9032252, upper bound: 113.9049453
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 213.60
Output dim: 13, lower bound: -113.9032252, upper bound: 114.0166456
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 213.60
Output dim: 13, lower bound: -113.9032252, upper bound: 113.9049453
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 213.60
Output dim: 13, lower bound: -113.9032252, upper bound: 114.0166456

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -103.3804550, 67.3323517, -103.5083237, 67.3855896, -170.7660370, 170.8406677
1: -49.9115715, 50.6353645, -49.9440842, 50.7020264, -100.6136017, 100.5794449
2: -48.5114326, 49.9821777, -48.5686569, 50.0762367, -98.5876694, 98.5508347
3: -52.0049744, 63.1405067, -52.1803894, 63.3022537, -115.3072281, 115.3208923
4: -65.5699844, 56.9141922, -65.6596222, 56.9783363, -122.5483093, 122.5738144
5: -55.1384125, 58.8983269, -55.2575645, 59.0302582, -114.1686554, 114.1558914
6: -86.3709106, 50.4740219, -86.4092255, 50.8020477, -137.1729584, 136.8832397
7: -67.2312012, 49.1431961, -67.3296814, 49.2323608, -116.4635620, 116.4728775
8: -78.5905762, 69.7108612, -78.7416077, 69.7137146, -148.3042755, 148.4524536
9: -61.6156693, 67.9926224, -61.6874199, 68.1469193, -129.7625885, 129.6800385
10: -91.6149597, 89.7570190, -91.7189789, 90.0131607, -181.6281128, 181.4759827
11: -83.5412292, 42.4999542, -83.6488953, 42.4646034, -126.0058136, 126.1488495
12: -60.6031761, 76.5093536, -60.3874474, 76.8690643, -137.4722443, 136.8968048
13: -68.0490723, 104.8724060, -67.8726349, 105.4065704, -173.4556274, 172.7450409
14: -116.8105850, 60.6578865, -117.0073929, 60.7023773, -177.5129700, 177.6652832
15: -66.7900009, 63.4204140, -67.0477600, 63.4586601, -130.2486572, 130.4681702
16: -97.4724274, 53.9804077, -97.5377121, 54.2480240, -151.7204590, 151.5181122
17: -109.7132034, 72.3529510, -109.8095932, 72.4445648, -182.1577759, 182.1625366
18: -91.0250244, 46.6109123, -91.4723511, 46.6821594, -137.7071533, 138.0832672
19: -67.0369415, 35.0107040, -67.2290192, 35.0139847, -102.0509262, 102.2397232
20: -65.7304535, 42.8233604, -65.9342728, 42.6553078, -108.3857574, 108.7576294
21: -84.9687805, 43.0694809, -85.1474915, 42.8544350, -127.8232117, 128.2169800
22: -75.0152969, 62.2066536, -75.4133072, 62.1241608, -137.1394653, 137.6199646
23: -66.8928833, 48.2839165, -67.1444550, 48.2383194, -115.1311951, 115.4283752
24: -85.6616058, 55.6765633, -86.0262146, 55.5193672, -141.1809387, 141.7027740
25: -63.0590172, 58.0234718, -63.2891045, 57.9092560, -120.9682770, 121.3125763
26: -90.6244202, 57.3801270, -91.1235352, 57.4143753, -148.0387726, 148.5036469
27: -99.0357513, 46.2565842, -99.5221863, 46.0687218, -145.1044617, 145.7787781
28: -66.5148163, 53.2623024, -66.8563690, 53.2890663, -119.8038788, 120.1186676
29: -78.0973206, 65.6475220, -78.4286423, 65.4755402, -143.5728302, 144.0761566
30: -83.2112732, 56.2807388, -83.4284973, 55.9514084, -139.1626892, 139.7092285
31: -89.7690811, 49.0573349, -89.9793625, 48.9996490, -138.7687073, 139.0366821
32: -76.8239441, 59.4542351, -76.8536987, 59.6484566, -136.4723969, 136.3079224
33: -113.4125061, 70.8479919, -113.3188324, 71.1152573, -184.5277710, 184.1668243
34: -90.2135468, 52.2108154, -90.2289581, 52.3281784, -142.5417175, 142.4397736
35: -88.9604492, 63.8061752, -88.7300415, 64.0125046, -152.9729614, 152.5362091
36: -88.0399017, 60.6052437, -87.8201141, 60.6963387, -148.7362366, 148.4253540
37: -136.3719482, 45.2801819, -136.2638245, 45.4668198, -181.8387604, 181.5440063
38: -109.7263718, 68.6187897, -109.5432892, 68.7935181, -178.5198669, 178.1620789
39: -122.0450974, 68.2149582, -121.9740677, 68.5849304, -190.6300354, 190.1890259
40: -112.7770233, 35.4366455, -112.8663635, 35.6053162, -148.3823242, 148.3030090
41: -86.1893921, 49.4777679, -86.1126556, 49.6667099, -135.8560944, 135.5904236
42: -58.2514496, 46.8504601, -58.3251228, 47.1563568, -105.4078064, 105.1755676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=458, inp2_unstable=459, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 821

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1587

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.8063491, upper bound: 114.0084382
time: 141.26 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.8950642, upper bound: 114.0084382
time: 124.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -103.6004715, 67.5211334, -103.5478134, 67.3956451, -170.9961243, 171.0689392
1: -50.0491982, 50.7837601, -49.9688110, 50.7095642, -100.7587585, 100.7525635
2: -48.6450272, 50.1412659, -48.5985031, 50.0834198, -98.7284470, 98.7397690
3: -52.1669426, 63.3483658, -52.2155342, 63.3115540, -115.4784851, 115.5639038
4: -65.7565002, 57.0883408, -65.6972809, 56.9859314, -122.7424316, 122.7856216
5: -55.2697639, 59.1098022, -55.2839890, 59.0408249, -114.3105927, 114.3937912
6: -86.4942017, 50.6102142, -86.4263229, 50.8201523, -137.3143616, 137.0365295
7: -67.4018555, 49.3296738, -67.3605194, 49.2419510, -116.6437988, 116.6901932
8: -78.7986450, 69.9204559, -78.7840195, 69.7260895, -148.5247345, 148.7044678
9: -61.7622108, 68.1406097, -61.7094994, 68.1553574, -129.9175720, 129.8501129
10: -91.7827301, 89.8723145, -91.7388611, 90.0282898, -181.8110199, 181.6111603
11: -83.8044357, 42.6515121, -83.6588974, 42.4956779, -126.3001099, 126.3104095
12: -60.7331963, 76.7039185, -60.4079361, 76.8891907, -137.6223907, 137.1118469
13: -68.4067993, 105.2703705, -67.9552460, 105.4171906, -173.8239899, 173.2256165
14: -117.0560684, 60.7870522, -117.0417404, 60.7101707, -177.7662354, 177.8287964
15: -66.9247894, 63.5188217, -67.0698547, 63.4737358, -130.3984985, 130.5886841
16: -97.6361847, 54.0916824, -97.5563889, 54.2590790, -151.8952484, 151.6480713
17: -110.0093079, 72.6472397, -109.8589630, 72.4543991, -182.4637146, 182.5061798
18: -91.3223114, 46.8603554, -91.4844971, 46.7379837, -138.0603027, 138.3448486
19: -67.2423630, 35.1554184, -67.2375259, 35.0481796, -102.2905273, 102.3929443
20: -65.8924713, 42.9680367, -65.9424286, 42.6859169, -108.5783844, 108.9104614
21: -85.2115707, 43.2492714, -85.1565933, 42.8953171, -128.1068878, 128.4058685
22: -75.2323380, 62.3620758, -75.4261169, 62.1577950, -137.3901367, 137.7881927
23: -67.1400146, 48.4773407, -67.1509705, 48.2808647, -115.4208832, 115.6283035
24: -85.8995667, 55.8727646, -86.0359039, 55.5644226, -141.4639893, 141.9086609
25: -63.2744598, 58.2191086, -63.2986069, 57.9517441, -121.2262039, 121.5177155
26: -90.9158401, 57.6071854, -91.1393204, 57.4658737, -148.3817139, 148.7465057
27: -99.3270111, 46.4887466, -99.5319519, 46.1236496, -145.4506531, 146.0206909
28: -66.7268524, 53.4477997, -66.8639679, 53.3307800, -120.0576324, 120.3117523
29: -78.3328018, 65.8064575, -78.4400330, 65.5110626, -143.8438721, 144.2464905
30: -83.3909073, 56.4627686, -83.4371872, 55.9887619, -139.3796539, 139.8999634
31: -90.0507660, 49.2623177, -89.9909515, 49.0467262, -139.0974884, 139.2532654
32: -76.9830933, 59.5890656, -76.8796234, 59.6616516, -136.6447449, 136.4686737
33: -113.6176453, 71.0760269, -113.3546829, 71.1273499, -184.7449951, 184.4307098
34: -90.3493118, 52.3105927, -90.2491760, 52.3353348, -142.6846466, 142.5597687
35: -89.1033478, 63.9577293, -88.7522049, 64.0231476, -153.1264954, 152.7099304
36: -88.1861649, 60.7001228, -87.8435364, 60.7043648, -148.8905334, 148.5436401
37: -136.5818939, 45.3994865, -136.2926331, 45.4775772, -182.0594788, 181.6921234
38: -109.9422226, 68.7459564, -109.5734940, 68.8043060, -178.7465210, 178.3194580
39: -122.3593063, 68.4647598, -122.0369873, 68.5908661, -190.9501648, 190.5017395
40: -112.9710693, 35.5767937, -112.8926010, 35.6132088, -148.5842743, 148.4693909
41: -86.3204193, 49.5522537, -86.1313477, 49.6779671, -135.9983826, 135.6835938
42: -58.3468399, 46.9457016, -58.3385277, 47.1710472, -105.5178833, 105.2842102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=458, inp2_unstable=459, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 821

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1587

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.8063491, upper bound: 114.0084382
time: 99.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.8063491, upper bound: 114.0084382
time: 105.57 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 207.78 seconds
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 207.78
Output dim: 13, lower bound: -113.8063491, upper bound: 114.0084382
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 207.78
Output dim: 13, lower bound: -113.8950642, upper bound: 114.0084382
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 207.78
Output dim: 13, lower bound: -113.8063491, upper bound: 114.0084382
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 207.78
Output dim: 13, lower bound: -113.8063491, upper bound: 114.0084382

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -103.3439713, 67.2002563, -103.2800064, 67.0093842, -170.3533478, 170.4802551
1: -49.8908043, 50.5858345, -49.8401871, 50.5558357, -100.4466400, 100.4260101
2: -48.4956818, 49.9262695, -48.4666748, 49.9100761, -98.4057617, 98.3929367
3: -51.9837723, 63.1028442, -52.0942764, 63.1810913, -115.1648407, 115.1971054
4: -65.5389557, 56.8899384, -65.5387268, 56.8978500, -122.4368057, 122.4286652
5: -55.1135750, 58.8696098, -55.1618156, 58.9327850, -114.0463562, 114.0314255
6: -86.3407440, 50.4254036, -86.2984009, 50.6453629, -136.9861145, 136.7238007
7: -67.2014847, 49.1267929, -67.2213669, 49.1704369, -116.3719177, 116.3481598
8: -78.5586777, 69.6793747, -78.6268082, 69.5976639, -148.1563416, 148.3061829
9: -61.5880470, 67.9539032, -61.5784836, 68.0238037, -129.6118469, 129.5323792
10: -91.5617752, 89.7313843, -91.5497818, 89.8963928, -181.4581604, 181.2811584
11: -83.4698334, 42.4808235, -83.4299316, 42.3605347, -125.8303680, 125.9107513
12: -60.5762901, 76.4551239, -60.2695885, 76.6930695, -137.2693634, 136.7247009
13: -68.0168762, 104.7275848, -67.6322937, 105.0035706, -173.0204468, 172.3598633
14: -116.7194595, 60.6445732, -116.7034912, 60.6004105, -177.3198700, 177.3480682
15: -66.7431564, 63.3996010, -66.8917160, 63.3817101, -130.1248627, 130.2913208
16: -97.4224014, 53.9145699, -97.3650436, 54.0437584, -151.4661560, 151.2796021
17: -109.6264648, 72.3250198, -109.5275650, 72.3424454, -181.9688721, 181.8525848
18: -90.9175415, 46.5940132, -91.1531143, 46.5420609, -137.4595947, 137.7471313
19: -66.9970398, 34.9966278, -67.1031036, 34.9470673, -101.9440918, 102.0997314
20: -65.6541290, 42.8078041, -65.7076340, 42.5372772, -108.1913910, 108.5154419
21: -84.8981400, 43.0545158, -84.9264145, 42.7412949, -127.6394348, 127.9809265
22: -74.9424286, 62.1861954, -75.1778564, 62.0354996, -136.9779205, 137.3640442
23: -66.8479309, 48.2621498, -67.0051727, 48.1349220, -114.9828491, 115.2673111
24: -85.6019440, 55.6561432, -85.8213348, 55.4324379, -141.0343781, 141.4774780
25: -62.9955826, 58.0023346, -63.0875435, 57.8057442, -120.8013153, 121.0898743
26: -90.5032425, 57.3593597, -90.7627258, 57.2316666, -147.7349091, 148.1220703
27: -98.9445267, 46.2416992, -99.2359161, 45.9455414, -144.8900452, 145.4776154
28: -66.4458008, 53.2459450, -66.6541748, 53.1691360, -119.6149368, 119.9001160
29: -78.0472717, 65.6226044, -78.2536087, 65.3796082, -143.4268646, 143.8762207
30: -83.0956879, 56.2650452, -83.0806046, 55.7888680, -138.8845520, 139.3456421
31: -89.7214813, 49.0374451, -89.8236847, 48.9163246, -138.6378021, 138.8611298
32: -76.7932434, 59.3806152, -76.6895752, 59.4151726, -136.2083893, 136.0701752
33: -113.3878860, 70.7522583, -113.1728134, 70.8416214, -184.2295074, 183.9250793
34: -90.1804047, 52.1811028, -90.1038208, 52.2350693, -142.4154663, 142.2849274
35: -88.9399109, 63.7281265, -88.5923309, 63.7848549, -152.7247620, 152.3204651
36: -88.0174713, 60.5191536, -87.6648102, 60.4557419, -148.4732056, 148.1839600
37: -136.3387756, 45.1728363, -136.0680695, 45.1566467, -181.4954224, 181.2409058
38: -109.6954651, 68.5057831, -109.3661041, 68.4679565, -178.1634216, 177.8718872
39: -122.0144348, 68.0603333, -121.7361832, 68.1541443, -190.1685791, 189.7965088
40: -112.7436676, 35.3591881, -112.7299500, 35.3771439, -148.1208191, 148.0891418
41: -86.1661758, 49.3910294, -85.9712067, 49.4033279, -135.5695038, 135.3622437
42: -58.2325058, 46.7821198, -58.2295761, 46.9446678, -105.1771698, 105.0116959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=458, inp2_unstable=458, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 821

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.6553854, upper bound: 113.9998548
time: 122.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.8000173, upper bound: 114.0021051
time: 124.04 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -103.3784561, 67.3277893, -103.5000763, 67.3659134, -170.7443695, 170.8278351
1: -49.9104805, 50.6334686, -49.9393845, 50.6939087, -100.6043854, 100.5728455
2: -48.5105400, 49.9798775, -48.5648689, 50.0669250, -98.5774689, 98.5447464
3: -52.0029716, 63.1388893, -52.1718597, 63.2960281, -115.2989578, 115.3107452
4: -65.5677185, 56.9123192, -65.6501999, 56.9702988, -122.5380173, 122.5625153
5: -55.1364746, 58.8970070, -55.2499199, 59.0248413, -114.1613159, 114.1469269
6: -86.3687134, 50.4701042, -86.3999176, 50.7854576, -137.1541748, 136.8700256
7: -67.2292633, 49.1418762, -67.3215637, 49.2267685, -116.4560318, 116.4634399
8: -78.5879822, 69.7084579, -78.7305908, 69.7036743, -148.2916412, 148.4390564
9: -61.6139183, 67.9902573, -61.6799278, 68.1391373, -129.7530518, 129.6701813
10: -91.6103363, 89.7556458, -91.6989136, 90.0076752, -181.6179810, 181.4545593
11: -83.5381393, 42.4989014, -83.6355896, 42.4602127, -125.9983521, 126.1344833
12: -60.6016579, 76.5072556, -60.3812675, 76.8602295, -137.4618835, 136.8885193
13: -68.0472794, 104.8694153, -67.8661499, 105.3938904, -173.4411621, 172.7355652
14: -116.8058167, 60.6567497, -116.9875107, 60.6974525, -177.5032654, 177.6442413
15: -66.7863312, 63.4189339, -67.0345459, 63.4524117, -130.2387390, 130.4534760
16: -97.4690094, 53.9782181, -97.5232086, 54.2391815, -151.7081909, 151.5014038
17: -109.7070618, 72.3510895, -109.7851562, 72.4368591, -182.1439209, 182.1362457
18: -91.0208893, 46.6098251, -91.4550018, 46.6777878, -137.6986694, 138.0648193
19: -67.0350800, 35.0099449, -67.2210541, 35.0108299, -102.0459137, 102.2309875
20: -65.7275696, 42.8223190, -65.9219666, 42.6511307, -108.3787003, 108.7442780
21: -84.9656601, 43.0686302, -85.1342239, 42.8509827, -127.8166351, 128.2028503
22: -75.0115433, 62.2055893, -75.3977814, 62.1197777, -137.1313171, 137.6033630
23: -66.8911133, 48.2827873, -67.1380768, 48.2337189, -115.1248322, 115.4208679
24: -85.6591263, 55.6755066, -86.0167007, 55.5147552, -141.1738892, 141.6921997
25: -63.0567093, 58.0223999, -63.2798500, 57.9047127, -120.9614258, 121.3022461
26: -90.6200714, 57.3787155, -91.1067047, 57.4087524, -148.0288239, 148.4854126
27: -99.0331726, 46.2556419, -99.5114822, 46.0648842, -145.0980530, 145.7671204
28: -66.5128784, 53.2613373, -66.8494339, 53.2853432, -119.7982178, 120.1107712
29: -78.0939560, 65.6463623, -78.4142532, 65.4705963, -143.5645447, 144.0606079
30: -83.2070007, 56.2798233, -83.4133453, 55.9478416, -139.1548157, 139.6931763
31: -89.7663269, 49.0562973, -89.9675446, 48.9952965, -138.7616272, 139.0238342
32: -76.8221130, 59.4505882, -76.8460541, 59.6329765, -136.4550781, 136.2966461
33: -113.4109116, 70.8439484, -113.3120346, 71.1013107, -184.5122223, 184.1559753
34: -90.2095032, 52.2092323, -90.2116318, 52.3213539, -142.5308533, 142.4208527
35: -88.9586945, 63.8017998, -88.7228241, 63.9969025, -152.9555969, 152.5246277
36: -88.0382690, 60.6020622, -87.8134460, 60.6825027, -148.7207642, 148.4154968
37: -136.3700714, 45.2762375, -136.2560425, 45.4496841, -181.8197479, 181.5322876
38: -109.7244339, 68.6135712, -109.5357056, 68.7751312, -178.4995575, 178.1492615
39: -122.0431442, 68.2106705, -121.9660645, 68.5707016, -190.6138306, 190.1767273
40: -112.7753143, 35.4334946, -112.8592834, 35.5923157, -148.3676300, 148.2927856
41: -86.1877441, 49.4743462, -86.1058350, 49.6526566, -135.8403931, 135.5801544
42: -58.2503319, 46.8476219, -58.3203773, 47.1446800, -105.3950119, 105.1679840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=458, inp2_unstable=458, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 576

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.6553854, upper bound: 113.9998548
time: 454.14 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.8887110, upper bound: 114.0021051
time: 101.18 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -103.5640182, 67.3890076, -103.3195190, 67.0194702, -170.5834961, 170.7085266
1: -50.0284653, 50.7342072, -49.8649330, 50.5633659, -100.5918274, 100.5991364
2: -48.6293144, 50.0853348, -48.4965401, 49.9172211, -98.5465393, 98.5818710
3: -52.1457443, 63.3106995, -52.1294327, 63.1904259, -115.3361511, 115.4401245
4: -65.7255173, 57.0640373, -65.5764084, 56.9054718, -122.6309891, 122.6404419
5: -55.2449188, 59.0810204, -55.1882553, 58.9433670, -114.1882782, 114.2692719
6: -86.4640045, 50.5615463, -86.3154831, 50.6634598, -137.1274719, 136.8770294
7: -67.3721619, 49.3132248, -67.2522354, 49.1800537, -116.5522079, 116.5654602
8: -78.7667999, 69.8889771, -78.6692657, 69.6100616, -148.3768311, 148.5582428
9: -61.7345619, 68.1018524, -61.6005325, 68.0322723, -129.7668304, 129.7023926
10: -91.7295227, 89.8466339, -91.5696335, 89.9114914, -181.6410217, 181.4162598
11: -83.7329712, 42.6324120, -83.4399185, 42.3916245, -126.1245880, 126.0723190
12: -60.7063828, 76.6496277, -60.2900543, 76.7131805, -137.4195557, 136.9396820
13: -68.3747330, 105.1254883, -67.7149506, 105.0141983, -173.3889313, 172.8404388
14: -116.9650650, 60.7737732, -116.7379227, 60.6081467, -177.5732117, 177.5116882
15: -66.8779221, 63.4980240, -66.9138260, 63.3967896, -130.2747192, 130.4118500
16: -97.5861740, 54.0258255, -97.3837357, 54.0548363, -151.6410065, 151.4095612
17: -109.9226074, 72.6193008, -109.5769348, 72.3523102, -182.2749176, 182.1962280
18: -91.2148590, 46.8434830, -91.1653137, 46.5978546, -137.8127136, 138.0087891
19: -67.2024536, 35.1413040, -67.1116180, 34.9812279, -102.1836853, 102.2529221
20: -65.8161469, 42.9524994, -65.7157745, 42.5678635, -108.3840103, 108.6682739
21: -85.1409683, 43.2343483, -84.9355469, 42.7821465, -127.9231110, 128.1698914
22: -75.1594696, 62.3416100, -75.1906662, 62.0691071, -137.2285767, 137.5322723
23: -67.0950851, 48.4555588, -67.0117111, 48.1774750, -115.2725525, 115.4672623
24: -85.8399353, 55.8523331, -85.8310089, 55.4775124, -141.3174438, 141.6833496
25: -63.2109871, 58.1979752, -63.0970650, 57.8482285, -121.0592117, 121.2950287
26: -90.7946625, 57.5864105, -90.7785645, 57.2831993, -148.0778503, 148.3649597
27: -99.2358093, 46.4738579, -99.2456894, 46.0004845, -145.2362976, 145.7195435
28: -66.6578217, 53.4314880, -66.6617889, 53.2108536, -119.8686752, 120.0932770
29: -78.2827148, 65.7815704, -78.2650070, 65.4151382, -143.6978455, 144.0465698
30: -83.2753143, 56.4471169, -83.0893250, 55.8262672, -139.1015625, 139.5364380
31: -90.0031662, 49.2424202, -89.8352890, 48.9633904, -138.9665527, 139.0777130
32: -76.9523849, 59.5154419, -76.7154999, 59.4283791, -136.3807678, 136.2309418
33: -113.5930939, 70.9802551, -113.2087631, 70.8537140, -184.4467773, 184.1889954
34: -90.3161774, 52.2807961, -90.1240540, 52.2423096, -142.5584717, 142.4048462
35: -89.0828400, 63.8797417, -88.6144791, 63.7955132, -152.8783569, 152.4942169
36: -88.1637497, 60.6139679, -87.6881714, 60.4637680, -148.6275177, 148.3021393
37: -136.5486755, 45.2920837, -136.0968933, 45.1674423, -181.7161255, 181.3889618
38: -109.9113388, 68.6328888, -109.3962860, 68.4787445, -178.3900757, 178.0291748
39: -122.3287125, 68.3101196, -121.7991257, 68.1600647, -190.4887695, 190.1092377
40: -112.9377060, 35.4992828, -112.7561646, 35.3850441, -148.3227539, 148.2554321
41: -86.2972107, 49.4655075, -85.9898911, 49.4146080, -135.7118225, 135.4553833
42: -58.3278885, 46.8773918, -58.2430000, 46.9593468, -105.2872314, 105.1203918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=458, inp2_unstable=458, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 821

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.6553854, upper bound: 113.9998548
time: 93.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.6553854, upper bound: 113.8859094
time: 1021.54 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 1117.06 seconds
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1117.06
Output dim: 13, lower bound: -113.6553854, upper bound: 113.9998548
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1117.06
Output dim: 13, lower bound: -113.8000173, upper bound: 114.0021051
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1117.06
Output dim: 13, lower bound: -113.6553854, upper bound: 113.9998548
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1117.06
Output dim: 13, lower bound: -113.8887110, upper bound: 114.0021051
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1117.06
Output dim: 13, lower bound: -113.6553854, upper bound: 113.9998548
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 1117.06
Output dim: 13, lower bound: -113.6553854, upper bound: 113.8859094
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1117.06
Output dim: 13, lower bound: -113.8063491, upper bound: 114.0084382
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=172.8270263671875
rel_dist={13: [-114.05422291258427, 114.05422290962252]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 13641.06 seconds
