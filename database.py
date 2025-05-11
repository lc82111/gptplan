import sqlite3
import pandas as pd
import textwrap
import numpy as np

class CervicalCancerDB:
    def __init__(self, db_name='.cervical_cancer.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()
        self.build_db()

    def build_db(self):
        # Add existing patients to the database
        # Name, PTV, Rectum, Bladder, Bowel Bag, Femoral Head L, Femoral Head R, R1, R2, R3, NT, Rectum Overlap, Bladder Overlap, Bowel Bag Overlap, Femoral Head Overlap L, Femoral Head Overlap R
        self.set_patient('A', 1328.1, 122.364, 295.476, 2598.17, 50.6408, 51.1471, 525.858, 880.024, 1035.4, 27465.3, 40.9525, 125.613, 123.655, 0.0, 0.0)
        self.set_patient('B', 1669.35, 50.9069, 397.177, 3038.23, 38.3351, 41.282, 503.483, 972.757, 1149.24, 27486.9, 30.0048, 178.172, 62.7014, 0.0, 0.0)
        self.set_patient('C', 1197.78, 89.0799, 410.282, 2465.73, 39.6107, 39.5749, 482.299, 845.793, 1034.28, 32775.4, 48.3272, 158.896, 100.207, 0.0, 0.0)
        self.set_patient('ZXY', 1078.5, 119.438, 512.74, 2865.26, 40.5572, 40.6883, 495.734, 875.075, 1058.64, 29059.1, 15.1085, 146.987, 120.029, 0.0, 0.0)
        self.set_patient('D', 1212.51, 122.053, 495.471, 2568.86, 31.1855, 34.3184, 440.497, 893.727, 1057.57, 26276.5, 63.4528, 156.969, 149.834, 0.0, 0.0)
        self.set_patient('E', 1220.88, 73.1872, 474.512, 1450.07, 39.1072, 37.953, 479.243, 857.996, 1053.37, 27219.8, 35.5101, 165.736, 171.561, 0.0, 0.0)
        self.set_patient('F', 1316.0, 91.7048, 293.499, 3688.02, 40.1399, 38.3828, 487.592, 916.565, 1108.36, 33724.6, 48.4369, 125.098, 118.605, 0.0, 0.0)
        self.set_patient('G', 1312.68, 37.5317, 502.152, 2758.19, 41.3153, 41.8303, 513.246, 925.086, 1117.11, 28726.3, 22.5638, 213.322, 154.666, 0.0, 0.0)
        self.set_patient('H', 1010.42, 33.7239, 173.912, 2024.78, 40.1635, 42.488, 516.522, 908.716, 1097.52, 27052.5, 12.1854, 70.1967, 50.6227, 0.0, 0.0)
        self.set_patient('I', 1030.37, 44.3578, 375.264, 2672.33, 45.8118, 46.3339, 471.143, 888.082, 1035.01, 28054.7, 11.4472, 72.5891, 87.9482, 0.0, 0.0)
        self.set_patient('Pat09', 1370.04, 165.078, 524.59, 2653.28, 44.6158, 44.8763, 475.504, 914.585, 1123.24, 34580.8, 81.6684, 141.144, 172.92, 0.0, 0.0)
        self.set_patient('Pat10', 924.313, 44.3313, 179.118, 2769.55, 45.4566, 47.3068, 490.892, 811.298, 981.786, 22270, 15.4113, 64.4204, 127.193, 0.0, 0.0)

        # Add Plan Optimization Parameters
        self.add_optim_parameters('B', [
            ('PTV', 'Min Dose', 4990, None, 90),
            ('PTV', 'Min DVH', 5040, 95, 90),
            ('PTV', 'Uniform Dose', 5140, None, 1),
            ('PTV', 'Max Dose', 5250, None, 70),
            ('rectum', 'Max DVH', 5000, 26, 40),
            ('rectum', 'Max DVH', 4000, 57, 30),
            ('rectum', 'Max EUD', 3400, None, 1),
            ('bladder', 'Max DVH', 5000, 38, 45),
            ('bladder', 'Max DVH', 4000, 25, 30),
            ('bladder', 'Max EUD', 3700, None, 1),
            ('bowel_bag', 'Max Dose', 5100, None, 30),
            ('bowel_bag', 'Max DVH', 4500, 3, 40),
            ('bowel_bag', 'Max DVH', 4000, 6, 30),
            ('bowel_bag', 'Max EUD', 1050, None, 1),
            ('femoral_head_l', 'Max DVH', 5000, 2, 40),
            ('femoral_head_l', 'Max DVH', 2000, 32, 20),
            ('femoral_head_l', 'Max EUD', 1700, None, 1),
            ('femoral_head_r', 'Max DVH', 5000, 2, 40),
            ('femoral_head_r', 'Max DVH', 2000, 35, 20),
            ('femoral_head_r', 'Max EUD', 1700, None, 1),
            ('r1', 'Max Dose', 4900, None, 30),
            ('r2', 'Max Dose', 4000, None, 30),
            ('r3', 'Max Dose', 3200, None, 30),
            ('nt', 'Max Dose', 3100, None, 30)
        ])

        self.add_optim_parameters('C', [
            ('PTV', 'Min Dose', 4990, None, 90),
            ('PTV', 'Min DVH', 5040, 95, 90),
            ('PTV', 'Uniform Dose', 5140, None, 1),
            ('PTV', 'Max Dose', 5300, None, 60),
            ('bladder', 'Max DVH', 5000, 36, 45),
            ('bladder', 'Max DVH', 4000, 52, 40),
            ('bladder', 'Max DVH', 3000, 70, 10),
            ('bladder', 'Max EUD', 3800, None, 1),
            ('rectum', 'Max DVH', 5000, 30, 40),
            ('rectum', 'Max DVH', 4000, 55, 40),
            ('rectum', 'Max DVH', 3000, 80, 10),
            ('rectum', 'Max EUD', 4000, None, 1),
            ('bowel_bag', 'Max Dose', 5180, None, 30),
            ('bowel_bag', 'Max DVH', 4500, 4, 40),
            ('bowel_bag', 'Max DVH', 4000, 7, 30),
            ('bowel_bag', 'Max EUD', 1000, None, 1),
            ('femoral_head_l', 'Max DVH', 5000, 1, 40),
            ('femoral_head_l', 'Max DVH', 2000, 17, 30),
            ('femoral_head_l', 'Max EUD', 1350, None, 1),
            ('femoral_head_r', 'Max DVH', 5000, 1, 40),
            ('femoral_head_r', 'Max DVH', 2000, 5, 30),
            ('femoral_head_r', 'Max EUD', 1000, None, 1),
            ('r1', 'Max Dose', 4900, None, 30),
            ('r2', 'Max Dose', 4300, None, 30),
            ('r3', 'Max Dose', 3300, None, 30),
            ('nt', 'Max Dose', 3100, None, 30),
        ])

        self.add_optim_parameters('D', [
            ('PTV', 'Min Dose', 4990, None, 90),
            ('PTV', 'Min DVH', 5040, 95, 90),
            ('PTV', 'Uniform Dose', 5140, None, 1),
            ('PTV', 'Max Dose', 5250, None, 60),
            ('bowel_bag', 'Max Dose', 5100, None, 20),
            ('bowel_bag', 'Max DVH', 4500, 6, 40),
            ('bowel_bag', 'Max EUD', 1500, None, 1),
            ('rectum', 'Max DVH', 5000, 38, 30),
            ('rectum', 'Max DVH', 4000, 57, 30),
            ('rectum', 'Max EUD', 3700, None, 1),
            ('bladder', 'Max DVH', 5000, 29, 30),
            ('bladder', 'Max DVH', 4000, 46, 20),
            ('bladder', 'Max EUD', 3300, None, 1),
            ('femoral_head_l', 'Max DVH', 5000, 1, 30),
            ('femoral_head_l', 'Max DVH', 2000, 2, 20),
            ('femoral_head_l', 'Max EUD', 730, None, 1),
            ('femoral_head_r', 'Max DVH', 5000, 1, 30),
            ('femoral_head_r', 'Max DVH', 2000, 2, 20),
            ('femoral_head_r', 'Max EUD', 780, None, 1),
            ('r1', 'Max Dose', 4900, None, 30),
            ('r2', 'Max Dose', 4100, None, 30),
            ('r3', 'Max Dose', 3300, None, 20),
            ('nt', 'Max Dose', 3100, None, 30)
        ])

        self.add_optim_parameters('E', [
            ('PTV', 'Min Dose', 4990, None, 90),
            ('PTV', 'Min DVH', 5040, 95, 90),
            ('PTV', 'Uniform Dose', 5140, None, 1),
            ('PTV', 'Max Dose', 5280, None, 60),
            ('rectum', 'Max DVH', 5000, 30, 35),
            ('rectum', 'Max DVH', 4000, 51, 30),
            ('rectum', 'Max EUD', 3200, None, 1),
            ('bladder', 'Max DVH', 5000, 32, 30),
            ('bladder', 'Max DVH', 4000, 53, 30),
            ('bladder', 'Max EUD', 3350, None, 1),
            ('bowel_bag', 'Max Dose', 5220, None, 20),
            ('bowel_bag', 'Max DVH', 5000, 6, 30),
            ('bowel_bag', 'Max DVH', 4500, 7, 45),
            ('bowel_bag', 'Max DVH', 4000, 14, 30),
            ('bowel_bag', 'Max EUD', 1700, None, 1),
            ('femoral_head_l', 'Max DVH', 5000, 1, 30),
            ('femoral_head_l', 'Max DVH', 2000, 3, 20),
            ('femoral_head_l', 'Max EUD', 830, None, 1),
            ('femoral_head_r', 'Max DVH', 5000, 1, 30),
            ('femoral_head_r', 'Max DVH', 2000, 3, 20),
            ('femoral_head_r', 'Max EUD', 900, None, 1),
            ('r1', 'Max Dose', 4900, None, 30),
            ('r2', 'Max Dose', 4000, None, 30),
            ('r3', 'Max Dose', 3100, None, 30),
            ('nt', 'Max Dose', 3000, None, 30)
        ])

        self.add_optim_parameters('F', [
            ('PTV', 'Min Dose', 4990, None, 90),
            ('PTV', 'Min DVH', 5040, 95, 100),
            ('PTV', 'Uniform Dose', 5160, None, 1),
            ('PTV', 'Max Dose', 5270, None, 60),
            ('rectum', 'Max DVH', 5000, 42, 40),
            ('rectum', 'Max DVH', 4000, 59, 30),
            ('rectum', 'Max EUD', 3800, None, 1),
            ('bladder', 'Max DVH', 5000, 40, 30),
            ('bladder', 'Max DVH', 4000, 52, 25),
            ('bladder', 'Max EUD', 3500, None, 1),
            ('bladder', 'Max DVH', 3000, 66, 10),
            ('bowel_bag', 'Max Dose', 5150, None, 20),
            ('bowel_bag', 'Max DVH', 5000, 2, 30),
            ('bowel_bag', 'Max DVH', 4500, 3, 40),
            ('bowel_bag', 'Max DVH', 4000, 4, 25),
            ('bowel_bag', 'Max EUD', 850, None, 1),
            ('femoral_head_l', 'Max DVH', 5000, 1, 30),
            ('femoral_head_l', 'Max DVH', 2000, 4, 20),
            ('femoral_head_l', 'Max EUD', 920, None, 1),
            ('femoral_head_r', 'Max DVH', 5000, 1, 30),
            ('femoral_head_r', 'Max DVH', 2000, 3, 20),
            ('femoral_head_r', 'Max EUD', 880, None, 1),
            ('r1', 'Max Dose', 4900, None, 30),
            ('r2', 'Max Dose', 4100, None, 30),
            ('r3', 'Max Dose', 3400, None, 30),
            ('nt', 'Max Dose', 3000, None, 30),
        ])

        self.add_optim_parameters('G', [
            ('PTV', 'Min Dose', 4990, None, 90),
            ('PTV', 'Min DVH', 5040, 95, 90),
            ('PTV', 'Uniform Dose', 5170, None, 1),
            ('PTV', 'Max Dose', 5250, None, 70),
            ('rectum', 'Max DVH', 5000, 19, 30),
            ('rectum', 'Max DVH', 4000, 44, 30),
            ('rectum', 'Max EUD', 3200, None, 1),
            ('rectum', 'Max DVH', 3000, 65, 20),
            ('bladder', 'Max DVH', 5000, 35, 30),
            ('bladder', 'Max DVH', 4000, 53, 25),
            ('bladder', 'Max EUD', 3500, None, 1),
            ('bowel_bag', 'Max Dose', 5200, None, 40),
            ('bowel_bag', 'Max DVH', 5000, 3, 30),
            ('bowel_bag', 'Max DVH', 4500, 4, 40),
            ('bowel_bag', 'Max DVH', 4000, 7, 30),
            ('bowel_bag', 'Max EUD', 970, None, 1),
            ('femoral_head_l', 'Max DVH', 5000, 1, 30),
            ('femoral_head_l', 'Max DVH', 3000, 15, 30),
            ('femoral_head_l', 'Max EUD', 1900, None, 1),
            ('femoral_head_r', 'Max DVH', 5000, 1, 30),
            ('femoral_head_r', 'Max DVH', 3000, 10, 30),
            ('femoral_head_r', 'Max EUD', 1800, None, 1),
            ('r1', 'Max Dose', 4900, None, 30),
            ('r2', 'Max Dose', 4200, None, 30),
            ('r3', 'Max Dose', 3300, None, 30),
            ('nt', 'Max Dose', 3100, None, 30),
        ])

        self.add_optim_parameters('H', [
            ('PTV', 'Min Dose', 4990, None, 100),
            ('PTV', 'Min DVH', 5040, 95, 95),
            ('PTV', 'Uniform Dose', 5160, None, 1),
            ('PTV', 'Max Dose', 5270, None, 60),
            ('rectum', 'Max DVH', 5000, 21, 30),
            ('rectum', 'Max DVH', 4000, 43, 20),
            ('rectum', 'Max EUD', 3200, None, 1),
            ('bladder', 'Max DVH', 5000, 32, 30),
            ('bladder', 'Max DVH', 3000, 78, 10),
            ('bladder', 'Max DVH', 4000, 56, 20),
            ('bladder', 'Max EUD', 3900, None, 1),
            ('bowel_bag', 'Max Dose', 5180, None, 30),
            ('bowel_bag', 'Max DVH', 5000, 2, 30),
            ('bowel_bag', 'Max DVH', 4500, 3, 40),
            ('bowel_bag', 'Max DVH', 4000, 4, 30),
            ('bowel_bag', 'Max EUD', 1050, None, 1),
            ('femoral_head_l', 'Max DVH', 5000, 1, 30),
            ('femoral_head_l', 'Max DVH', 3000, 2, 30),
            ('femoral_head_l', 'Max EUD', 1100, None, 1),
            ('femoral_head_r', 'Max DVH', 5000, 1, 30),
            ('femoral_head_r', 'Max DVH', 3000, 2, 30),
            ('femoral_head_r', 'Max EUD', 1150, None, 1),
            ('r1', 'Max Dose', 4900, None, 30),
            ('r2', 'Max Dose', 4200, None, 30),
            ('r3', 'Max Dose', 3300, None, 30),
            ('nt', 'Max Dose', 3100, None, 30),
        ])

        self.add_optim_parameters('I', [
            ('PTV', 'Min Dose', 4990, None, 90),
            ('PTV', 'Min DVH', 5040, 95, 90),
            ('PTV', 'Uniform Dose', 5140, None, 1),
            ('PTV', 'Max Dose', 5230, None, 60),
            ('rectum', 'Max DVH', 5000, 14, 30),
            ('rectum', 'Max DVH', 4000, 37, 30),
            ('rectum', 'Max EUD', 3200, None, 1),
            ('bladder', 'Max DVH', 5000, 15, 30),
            ('bladder', 'Max DVH', 4000, 33, 30),
            ('bladder', 'Max EUD', 2900, None, 1),
            ('bowel_bag', 'Max Dose', 5230, None, 40),
            ('bowel_bag', 'Max EUD', 1050, None, 1),
            ('bowel_bag', 'Max DVH', 5000, 3, 30),
            ('bowel_bag', 'Max DVH', 4500, 4, 40),
            ('bowel_bag', 'Max DVH', 4000, 5, 30),
            ('femoral_head_l', 'Max DVH', 5000, 1, 30),
            ('femoral_head_l', 'Max DVH', 3000, 5, 30),
            ('femoral_head_l', 'Max EUD', 1100, None, 1),
            ('femoral_head_r', 'Max DVH', 5000, 1, 30),
            ('femoral_head_r', 'Max DVH', 3000, 3, 30),
            ('femoral_head_r', 'Max EUD', 970, None, 1),
            ('r1', 'Max Dose', 4900, None, 30),
            ('r2', 'Max Dose', 4200, None, 30),
            ('r3', 'Max Dose', 3200, None, 30),
            ('nt', 'Max Dose', 3100, None, 30)
        ])

        self.add_optim_parameters('Pat09', [
            ('PTV', 'Min Dose', 4990, None, 90),
            ('PTV', 'Min DVH', 5040, 96, 100),
            ('PTV', 'Uniform Dose', 5140, None, 1),
            ('PTV', 'Max Dose', 5270, None, 70),
            ('rectum', 'Max EUD', 3800, None, 1),
            ('rectum', 'Max DVH', 5000, 40, 30),
            ('rectum', 'Max DVH', 4000, 56, 30),
            ('rectum', 'Max DVH', 3000, 67, 10),
            ('rectum', 'Max DVH', 2000, 80, 2),
            ('bladder', 'Max DVH', 5000, 22, 30),
            ('bladder', 'Max DVH', 4000, 37, 30),
            ('bladder', 'Max DVH', 3000, 46, 10),
            ('bladder', 'Max EUD', 2900, None, 1),
            ('bowel_bag', 'Max Dose', 5150, None, 40),
            ('bowel_bag', 'Max EUD', 1000, None, 1),
            ('bowel_bag', 'Max DVH', 5000, 4, 30),
            # ('bowel_bag', 'Max DVH', 4400, 3, 55),
            ('bowel_bag', 'Max DVH', 4500, 3, 55),
            ('bowel_bag', 'Max DVH', 4000, 8, 30),
            ('femoral_head_l', 'Max DVH', 5000, 1, 30),
            ('femoral_head_l', 'Max DVH', 2000, 2, 20),
            ('femoral_head_l', 'Max EUD', 930, None, 1),
            ('femoral_head_r', 'Max DVH', 5000, 1, 30),
            ('femoral_head_r', 'Max DVH', 2000, 2, 20),
            ('femoral_head_r', 'Max EUD', 800, None, 1),
            ('r1', 'Max Dose', 4900, None, 30),
            ('r2', 'Max Dose', 4100, None, 30),
            ('r3', 'Max Dose', 3400, None, 30),
            ('nt', 'Max Dose', 3100, None, 30)
        ])

        self.add_optim_parameters('Pat10', [
            ('PTV', 'Min Dose', 4990, None, 95),
            ('PTV', 'Min DVH', 5040, 96, 100),
            ('PTV', 'Uniform Dose', 5160, None, 1),
            ('PTV', 'Max Dose', 5280, None, 60),
            ('rectum', 'Max EUD', 3400, None, 1),
            ('rectum', 'Max DVH', 5000, 23, 30),
            ('rectum', 'Max DVH', 4000, 44, 30),
            ('rectum', 'Max DVH', 3000, 58, 10),
            ('bladder', 'Max DVH', 5000, 33, 30),
            ('bladder', 'Max DVH', 4000, 53, 30),
            ('bladder', 'Max DVH', 3000, 71, 10),
            ('bladder', 'Max EUD', 3600, None, 1),
            ('bowel_bag', 'Max Dose', 5210, None, 40),
            ('bowel_bag', 'Max EUD', 1200, None, 1),
            ('bowel_bag', 'Max DVH', 5000, 2, 30),
            ('bowel_bag', 'Max DVH', 4500, 4, 40),
            ('bowel_bag', 'Max DVH', 4000, 7, 20),
            ('femoral_head_L', 'Max DVH', 5000, 1, 30),
            ('femoral_head_L', 'Max DVH', 2000, 3, 20),
            ('femoral_head_L', 'Max EUD', 920, None, 1),
            ('femoral_head_R', 'Max DVH', 5000, 1, 30),
            ('femoral_head_R', 'Max DVH', 2000, 3, 20),
            ('femoral_head_R', 'Max EUD', 1120, None, 1),
            ('r1', 'Max Dose', 4900, None, 30),
            ('r2', 'Max Dose', 4100, None, 30),
            ('r3', 'Max Dose', 3400, None, 30),
            ('nt', 'Max Dose', 3100, None, 30)
        ])

        ## add trajectories
        self.add_trajectory('C', """\
A1:
PTV: MinD 4990 (w:90), MinDVH 5040/95% (w:90), UniD 5140 (w:1), MaxD 5300 (w:60)
Blad: MaxDVH 5000/36% (w:45), MaxDVH 4000/52% (w:40), MaxEUD 3800 (w:1)
Rect: MaxDVH 5000/30% (w:40), MaxDVH 4000/55% (w:40), MaxEUD 4000 (w:1)
Bowl: MaxD 5180 (w:30), MaxDVH 4500/4% (w:40), MaxEUD 1000 (w:1)
FemL: MaxDVH 5000/1% (w:40), MaxDVH 2000/17% (w:30), MaxEUD 1350 (w:1)
FemR: MaxDVH 5000/1% (w:40), MaxDVH 2000/5% (w:30), MaxEUD 1000 (w:1)
->
S1:
PTV: D95 5000, MeanD 5303.2, MaxD 5777.6
Blad: MeanD 4257.2, V50 39%, V30 83%
Rect: MeanD 4379, V50 41%, V30 93%
Bowl: MaxD 5685.4, MeanD 1099.4, V45 6%
FemL: MeanD 1290.9, V20 12%
FemR: MeanD 968.6, V20 5%

A2 (Changes only):
Blad: MaxDVH 3000/70% (w:10) [Added]
Rect: MaxDVH 3000/80% (w:10) [Added]
->
S2:
PTV: D95 5000 (=), MeanD 5303.6 (+0.4), MaxD 5723.1 (-54.5)
Blad: MeanD 4180.3 (-76.9), V50 39% (=), V30 77% (-6%)
Rect: MeanD 4314.9 (-64.1), V50 42% (+1%), V30 86% (-7%)
Bowl: MaxD 5682.4 (-3), MeanD 1094 (-5.4), V45 6% (=)
FemL: MeanD 1285.9 (-5), V20 12% (=)
FemR: MeanD 969.9 (+1.3), V20 5% (=)
""")

        self.add_trajectory('D', """\
A1:
PTV: MinD 4990 (w:90), MinDVH 5040/95% (w:90), UniD 5160 (w:1), MaxD 5300 (w:50)
Bowl: MaxD 5300 (w:20), MaxDVH 4500/6% (w:40), MaxEUD 1600 (w:1)
Rect: MaxDVH 5000/38% (w:30), MaxDVH 4000/58% (w:20), MaxEUD 3900 (w:1)
Blad: MaxDVH 5000/30% (w:30), MaxDVH 4000/47% (w:20), MaxEUD 3400 (w:1)
FemL: MaxDVH 5000/2% (w:30), MaxDVH 2000/8% (w:20), MaxEUD 950 (w:1)
FemR: MaxDVH 5000/2% (w:30), MaxDVH 2000/8% (w:20), MaxEUD 1000 (w:1)
->
S1:
PTV: D95 5062, MeanD 5275.9, MaxD 5625.1
Bowl: MaxD 5560.6, MeanD 1748, V45 8%
Rect: MeanD 4370.7, V50 45%
Blad: MeanD 3613.7, V50 32%
FemL: MeanD 916.5, V20 3%
FemR: MeanD 975.2, V20 4%

A2 (Changes only):
Bowl: MaxD 5250 (-50), MaxEUD 1500 (-100)
Rect: MaxDVH 4000/57% (-1%, w:+10), MaxEUD 3800 (-100)
Blad: MaxDVH 4000/46% (-1%), MaxEUD 3300 (-100)
FemL: MaxDVH 5000/1% (-1%), MaxDVH 2000/3% (-5%), MaxEUD 800 (-150)
FemR: MaxDVH 5000/1% (-1%), MaxDVH 2000/3% (-5%), MaxEUD 850 (-150)
->
S2:
PTV: D95 5062 (=), MeanD 5303.9 (+28), MaxD 5738.4 (+113.3)
Bowl: MaxD 5582.1 (+21.5), MeanD 1710.4 (-37.6), V45 8% (=)
Rect: MeanD 4344.1 (-26.6), V50 45% (=)
Blad: MeanD 3595.8 (-17.9), V50 32% (=)
FemL: MeanD 791.7 (-124.8), V20 0% (-3%)
FemR: MeanD 856.3 (-118.9), V20 1% (-3%)

A3 (Changes only):
Bowl: MaxD 5150 (-100)
Blad: MaxDVH 5000/29% (-1%)
FemL: MaxDVH 2000/2% (-1%), MaxEUD 730 (-70)
FemR: MaxDVH 2000/2% (-1%), MaxEUD 800 (-50)
->
S3:
PTV: D95 5050 (-12), MeanD 5303.7 (-0.2), MaxD 5730.6 (-7.8)
Bowl: MaxD 5591.6 (+9.5), MeanD 1702.6 (-7.8), V45 8% (=)
Rect: MeanD 4260 (-84.1), V50 45% (=)
Blad: MeanD 3592.8 (-3), V50 32% (=)
FemL: MeanD 763.4 (-28.3), V20 0% (=)
FemR: MeanD 842.3 (-14), V20 1% (=)
""")

        self.add_trajectory('E', """\
A1:
PTV: MinD 4990 (w:90), MinDVH 5040/95% (w:90), UniD 5150 (w:1), MaxD 5280 (w:60)
Blad: MaxDVH 5000/32% (w:30), MaxDVH 4000/53% (w:30), MaxEUD 3450 (w:1)
Rect: MaxDVH 5000/30% (w:35), MaxDVH 4000/52% (w:30), MaxEUD 3500 (w:1)
Bowl: MaxD 5220 (w:20), MaxDVH 5000/6% (w:30), MaxDVH 4500/8% (w:45), MaxDVH 4000/14% (w:30), MaxEUD 1700 (w:1)
FemL: MaxDVH 5000/1% (w:30), MaxDVH 2000/3% (w:20), MaxEUD 900 (w:1)
FemR: MaxDVH 5000/1% (w:30), MaxDVH 2000/3% (w:20), MaxEUD 950 (w:1)
->
S1:
PTV: D95 5062, MeanD 5305.3, MaxD 5635.5
Blad: MeanD 4067.5, V50 38%, V40 58%
Rect: MeanD 4035.8, V50 40%, V40 57%
Bowl: MaxD 5513.2, MeanD 2080.6, V45 13%
FemL: MeanD 870, V20 2% 
FemR: MeanD 430.6, V20 2%

A2 (Changes only):
Blad: MaxEUD 3350 (-100)
Rect: MaxDVH 4000/51% (-1%), MaxEUD 3200 (-300)
Bowl: MaxDVH 4500/7% (-1%)
FemL: MaxEUD 830 (-70)
FemR: MaxEUD 900 (-50)
->
S2:
PTV: D95 5050 (-12), MeanD 5304.7 (-0.6), MaxD 5705.1 (+69.6)
Blad: MeanD 4035.9 (-31.6), V50 37% (-1%), V40 57% (-1%)
Rect: MeanD 3958.6 (-77.2), V50 38% (-2%), V40 56% (-1%)
Bowl: MaxD 5556.4 (+43.2), MeanD 2077.8 (-2.8), V45 13% (=)
FemL: MeanD 811.8 (-58.2), V20 1% (-1%)
FemR: MeanD 914.9 (+484.3), V20 1% (-1%)
""")
        
        self.add_trajectory('F', """\
A1:
PTV: MinD 4990 (w:90), MinDVH 5040/95% (w:100), UniD 5160 (w:1), MaxD 5270 (w:60)
Blad: MaxDVH 5000/40% (w:30), MaxDVH 4000/52% (w:25), MaxEUD 3500 (w:1)
Rect: MaxDVH 5000/42% (w:40), MaxDVH 4000/59% (w:30), MaxEUD 3800 (w:1)
Bowl: MaxD 5150 (w:20), MaxDVH 4500/3% (w:40), MaxEUD 850 (w:1)
FemL: MaxDVH 5000/1% (w:30), MaxDVH 2000/4% (w:20), MaxEUD 920 (w:1)
FemR: MaxDVH 5000/1% (w:30), MaxDVH 2000/3% (w:20), MaxEUD 880 (w:1)
->
S1:
PTV: D95 5050, MeanD 5305, MaxD 5696
Blad: MeanD 4064, V50 43%, V40 58%
Rect: MeanD 4357, V50 46%, V40 66%
Bowl: MaxD 5562, MeanD 981, V45 4%
FemL: MeanD 941, V20 3%
FemR: MeanD 901, V20 1%

A2 (Changes only):
Blad: MaxDVH 3000/66% (w:10) [Added]
Rect: MaxDVH 3000/75% (w:10) [Added]
->
S2:
PTV: D95 5050 (=), MeanD 5304 (-1), MaxD 5713 (+17)
Blad: MeanD 3984 (-80), V50 42% (-1), V40 56% (-2)
Rect: MeanD 4247 (-110), V50 46% (=), V40 64% (-2)
Bowl: MaxD 5575 (+13), MeanD 975 (-7), V45 4% (=)
FemL: MeanD 921 (-20), V20 2% (-1)
FemR: MeanD 892 (-9), V20 1% (=)
""")

        self.add_trajectory("G", """\
A1:
PTV: MinD 4990 (w:90), MinDVH 5040/95% (w:90), UniD 5170 (w:1), MaxD 5250 (w:70)
Blad: MaxDVH 5000/35% (w:30), MaxDVH 4000/53% (w:25), MaxEUD 3500 (w:1)
Rect: MaxDVH 5000/19% (w:30), MaxDVH 4000/44% (w:30), MaxEUD 3200 (w:1)
Bowl: MaxD 5220 (w:35), MaxDVH 5000/3% (w:30), MaxDVH 4500/4% (w:40), MaxDVH 4000/7% (w:30), MaxEUD 970 (w:1)
FemL: MaxDVH 5000/1% (w:30), MaxDVH 3000/15% (w:30), MaxEUD 1800 (w:1)
FemR: MaxDVH 5000/1% (w:30), MaxDVH 3000/10% (w:30), MaxEUD 1800 (w:1)
Patient: MaxD 5450 (w:80)
->
S1:
PTV: D95 5074, MeanD 5302.2, MaxD 5631.5
Blad: MeanD 4024.3, V50 43%
Rect: MeanD 4013.4, V50 26%
Bowl: MaxD 5556.5, MeanD 1198.7, V45 7%, V40 8%
FemL: MeanD 2641, V20 97%
FemR: MeanD 1837.5, V20 33%

A2 (Changes only):
Bowl: MaxD 5200 (-20, w:=)
Rect: MaxDVH 3000/70% (w:20) [Added]
FemL: MaxEUD 1900 (+100)
Patient: MaxD 5400 (-50, w:+10)
->
S2:
PTV:  D95 5074 (=), MeanD 5302.8 (+0.6), MaxD 5665 (+33.5)
Blad: MeanD 3991 (-33.3), V50 43% (=)
Rect: MeanD 3983.3 (-30.1), V50 27% (+1%)
Bowl: MaxD 5519.4 (-37.1), MeanD 1177.1 (-21.6), V45 7% (=), V40 8% (=)
FemL: MeanD 1939 (-702), V20 41% (-56%)
FemR: MeanD 1857.6 (+20.1), V20 33% (=)

A3 (Changes only):
Bowl: MaxD 5200 (=, w:+5)
Rect: MaxDVH 3000/65% (-5%)
FemL: MaxEUD 1900 (=)
Patient: MaxD 5400 (=)
->
S3:
PTV: D95 5038 (-36), MeanD 5271.8 (-31), MaxD 5652.5 (-12.5)
Blad: MeanD 3960.1 (-30.9), V50 43% (=)
Rect: MeanD 3798.7 (-184.6), V50 28% (+1%)
Bowl: MaxD 5545.5 (+26.1), MeanD 1156.8 (-20.3), V45 7% (=), V40 8% (=)
FemL: MeanD 1903.6 (-35.4), V20 37% (-4%)
FemR: MeanD 1835.2 (-22.4), V20 33% (=)
""")

        self.add_trajectory('H', """\
A1:
PTV: MinD 4990 (w:100), MinDVH 5040/95% (w:95), UniD 5160 (w:1), MaxD 5270 (w:60)
Blad: MaxDVH 5000/33% (w:30), MaxDVH 4000/56% (w:20), MaxEUD 3900 (w:1), MaxDVH 3000/80% (w:10)
Rect: MaxDVH 5000/21% (w:30), MaxDVH 4000/45% (w:20), MaxEUD 3200 (w:1)
Bowl: MaxD 5180 (w:30), MaxDVH 5000/2% (w:30), MaxDVH 4500/3% (w:40), MaxDVH 4000/4% (w:30), MaxEUD 1050 (w:1)
FemL: MaxDVH 5000/1% (w:30), MaxDVH 3000/2% (w:30), MaxEUD 1100 (w:1)
FemR: MaxDVH 5000/1% (w:30), MaxDVH 3000/2% (w:30), MaxEUD 1150 (w:1)
Patient: MaxD 5400 (w:90)
->
S1:
PTV: D95 5026, MeanD 5304.7, MaxD 5917.4
Blad: MeanD 4473.1, V50 40%, V40 67%
Rect: MeanD 3809.7, V50 26%
Bowl: MaxD 5543.6, MeanD 1176.7, V45 4%, V40 6%
FemL: MeanD 1109.4, V20 5%
FemR: MeanD 1255.5, V20 13%

A2 (Changes only):
Blad: MaxDVH 3000/78% (-2%)
Rect: MaxDVH 4000/43% (-2%)
->
S2:
PTV: D95 5014 (-12), MeanD 5306.4 (+1.7), MaxD 5813.4 (-104)
Blad: MeanD 4347.7 (-125.4), V50 40% (=), V40 63% (-4%)
Rect: MeanD 3784.3 (-25.4), V50 25% (-1%)
Bowl: MaxD 5527.3 (-16.3), MeanD 1153.1 (-23.6), V45 4% (=), V40 6% (=)
FemL: MeanD 1112.8 (+3.4), V20 5% (=)
FemR: MeanD 1260.2 (+4.7), V20 13% (=)
""")

        self.add_trajectory('Pat09', """\
A1:
PTV: MinD 4990 (w:90), MinDVH 5040/96% (w:100), UniD 5160 (w:1), MaxD 5270 (w:70)
Blad: MaxDVH 5000/22% (w:30), MaxDVH 4000/37% (w:30), MaxEUD 2900 (w:1), MaxDVH 3000/46% (w:10)
Rect: MaxDVH 5000/40% (w:30), MaxDVH 4000/56% (w:30), MaxEUD 3800 (w:1), MaxDVH 3000/67% (w:10)
Bowl: MaxD 5150 (w:40), MaxDVH 4400/3% (w:55), MaxEUD 1000 (w:1)
FemL: MaxDVH 5000/1% (w:30), MaxDVH 2000/2% (w:20), MaxEUD 930 (w:1)
FemR: MaxDVH 5000/1% (w:30), MaxDVH 2000/2% (w:20), MaxEUD 800 (w:1)
Patient: MaxD 5500 (w:100)
R3: MaxD 3300 (w:30)
->
S1:
PTV: D95 5008, MeanD 5304.9, MaxD 6055.1
Blad: MeanD 3179, V50 28%, V40 40%
Rect: MeanD 4136, V50 46%, V40 60%
Bowl: MaxD 5624, MeanD 1248, V45 8%
FemL: MaxD 2843.5, MeanD 944.9, V20 2%
FemR: MaxD 2114.3, MeanD 791.6, V20 0%

A2 (Changes only):
Patient: MaxD 5400 (-100)
R3: MaxD 3400 (+100)
->
S2:
PTV: D95 5014 (+6), MeanD 5307 (+2.1), MaxD 5834.5 (-220.6)
Blad: MeanD 3174.7 (-4.3), V50 28% (=), V40 40% (=)
Rect: MeanD 4124.5 (-11.5), V50 47% (+1%), V40 60% (=)
Bowl: MaxD 5656.7 (+32.7), MeanD 1246.3 (-1.7), V45 8% (=)
FemL: MeanD 951.8 (+6.9), V20 2% (=)
FemR: MeanD 782.6 (-9), V20 0% (=)
""")
        
    def create_tables(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS patient_volumes
                              (name TEXT PRIMARY KEY, ptv REAL, rectum REAL, bladder REAL, bowel_bag REAL, femoral_head_l REAL, femoral_head_r REAL, r1 REAL, r2 REAL, r3 REAL, nt REAL)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS overlapping_volumes
                              (name TEXT PRIMARY KEY, rectum REAL, bladder REAL, bowel_bag REAL, femoral_head_l REAL, femoral_head_r REAL)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS plan_optimization_parameters
                        (patient_name TEXT, roi_name TEXT, objective_type TEXT, target_cgy REAL, percent_volume REAL, weight INTEGER, 
                        PRIMARY KEY (patient_name, roi_name, objective_type, target_cgy))''')
        self.cursor.execute(''' CREATE TABLE IF NOT EXISTS Optimization_Trajectory
                            (patient_name TEXT PRIMARY KEY, trajectory TEXT)''')

    def set_patient(self, name, ptv, rectum, bladder, bowel_bag, femoral_head_l, femoral_head_r, r1, r2, r3, nt, rectum_overlap, bladder_overlap, bowel_bag_overlap, femoral_head_overlap_l, femoral_head_overlap_r):
        patient_volumes = (name, ptv, rectum, bladder, bowel_bag, femoral_head_l, femoral_head_r, r1, r2, r3, nt)
        overlapping_volumes = (name, rectum_overlap, bladder_overlap, bowel_bag_overlap, femoral_head_overlap_l, femoral_head_overlap_r)
        self.cursor.execute("INSERT OR REPLACE INTO patient_volumes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", patient_volumes)
        self.cursor.execute("INSERT OR REPLACE INTO overlapping_volumes VALUES (?, ?, ?, ?, ?, ?)", overlapping_volumes)
        self.conn.commit()

    def get_patient(self, name):
        self.cursor.execute("SELECT * FROM patient_volumes WHERE name = ?", (name,))
        patient_volumes = self.cursor.fetchone()
        self.cursor.execute("SELECT * FROM overlapping_volumes WHERE name = ?", (name,))
        overlapping_volumes = self.cursor.fetchone()
        return patient_volumes, overlapping_volumes

    def add_optim_parameters(self, patient_name, parameters):
        for roi_name, objective_type, target_cgy, percent_volume, weight in parameters:
            self.cursor.execute(
                '''INSERT OR REPLACE INTO plan_optimization_parameters VALUES (?, ?, ?, ?, ?, ?)''',
                (patient_name, roi_name, objective_type, target_cgy, percent_volume, weight)
            )
        self.conn.commit()

    def get_optim_parameters(self, patient_name):
        self.cursor.execute("SELECT * FROM plan_optimization_parameters WHERE patient_name = ?", (patient_name,))
        return self.cursor.fetchall()

    def add_trajectory(self, patient_name, trajector):
        self.cursor.execute(
            ''' INSERT OR REPLACE INTO Optimization_Trajectory VALUES (?, ?)''',
            (patient_name, trajector)
        )
        self.conn.commit()

    def get_trajectory(self, patient_name):
        self.cursor.execute("SELECT * FROM Optimization_Trajectory WHERE patient_name = ?", (patient_name,))
        traj = self.cursor.fetchone()
        if traj:
            return traj[1]
        else:
            return None

    def get_all_patient_names(self):
        self.cursor.execute("SELECT DISTINCT name FROM patient_volumes")
        return [row[0] for row in self.cursor.fetchall()] 

    def close(self):
        self.conn.close()

    @staticmethod    
    def unit_test():
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        db = CervicalCancerDB()
        import pdb; pdb.set_trace()
        print(db.get_trajectory('C'))
        db.close()

if __name__ == '__main__':
    CervicalCancerDB.unit_test()
