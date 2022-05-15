use FeatureCountSets;
impl FeatureCountSets {
pub fn load_trained_score_params() -> FeatureCountSets {
FeatureCountSets {
hairpin_loop_length_counts: [-6.0238533, -3.1389818, 0.38533026, 2.1720896, 1.8843772, -0.62552136, -0.12157995, 0.5413163, -0.75342786, -0.20108281, -0.32872367, -0.0561095, -0.9503774, -0.05380082, -0.12631105, 0.16941397, -0.11138695, 0.15286306, -0.098552205, -0.35821345, -0.12076626, -0.29788, -0.34560928, -0.19744019, -0.05542488, -0.04365503, 0.044056997, 0.065203995, 0.09561847, 0.16636088, 0.23237062],
bulge_loop_length_counts: [-2.3882532, -0.8922529, -0.9074972, -0.83939826, -0.43457258, -0.5676421, 0.19955197, 0.75440043, -0.6034326, -0.72036195, -0.5146054, -0.36240995, -0.26234442, -0.16022441, -0.08703238, -0.03147672, -0.011373591, 0.02967005, 0.04727341, -0.04327206, -0.01812955, -0.07821154, -0.07113522, -0.057817508, -0.046417706, -0.035673406, -0.0268063, -0.018199483, -0.01053119, -0.0051599485],
interior_loop_length_counts: [-0.3785861, -0.34463465, -0.38424054, -0.32349253, -0.27461663, -0.07543881, -0.068282284, -0.02588837, -0.20119163, -0.28651056, -0.36605936, -0.31058356, -0.05436908, -0.14074102, -0.061635822, -0.09940933, -0.024019012, 0.0074743954, 0.016037783, -0.078829035, -0.14283758, -0.15055093, -0.07438177, -0.08500832, -0.056958348, -0.04640229, -0.00069540535, 0.002908244, 0.0056808996],
interior_loop_length_counts_symm: [-0.4807466, -0.36140993, -0.2592099, -0.23910679, 0.13888596, -0.66026205, -0.30515635, -0.033139337, -0.35413513, -0.21805033, -0.124331266, -0.1563186, -0.0862115, -0.04635863, -0.022455163],
interior_loop_length_counts_asymm: [-2.120977, -0.55836993, -0.5811671, -0.617122, -0.30897847, -0.1184171, -0.21255967, -0.31613594, -0.3167048, -0.09168293, -0.22111066, -0.14153405, -0.21701266, -0.17319217, -0.15649141, -0.10441, -0.06993899, -0.04130232, -0.015897017, 0.013697211, 0.04129106, 0.03591501, 0.028214168, 0.016542759, 0.025742332, 0.033778027, 0.039715424, -0.0025448848],
stack_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.10176581], [0.0, 0.0, 0.41241503, 0.0], [0.0, 0.66305345, 0.0, -0.10245719], [0.22982612, 0.0, 0.15227315, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.45276985], [0.0, 0.0, 0.80392575, 0.0], [0.0, 0.47000247, 0.0, -0.17893703], [0.66305345, 0.0, 0.47222888, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.50122046], [0.0, 0.0, 0.4831877, 0.0], [0.0, 0.80392575, 0.0, 0.21687907], [0.41241503, 0.0, 0.47725314, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, -0.04886642], [0.0, 0.0, 0.47725314, 0.0], [0.0, 0.47222888, 0.0, 0.18464492], [0.15227315, 0.0, -0.28943086, 0.0]]], [[[0.0, 0.0, 0.0, 0.38777992], [0.0, 0.0, 0.50122046, 0.0], [0.0, 0.45276985, 0.0, -0.11459199], [0.10176581, 0.0, -0.04886642, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, -0.11459199], [0.0, 0.0, 0.21687907, 0.0], [0.0, -0.17893703, 0.0, 0.11847955], [-0.10245719, 0.0, 0.18464492, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
terminal_mismatch_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.18313888, -0.1175684, -0.44307387, -0.60838884], [0.00257943, 0.08366669, -0.21838029, -0.39363334], [0.519904, -0.3467602, -0.4012573, -0.76418453], [-0.017274963, 0.26986718, -0.090530194, 0.33457947]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.07275507, -0.25581446, -0.6656412, -0.37776688], [0.1091483, -0.16515505, -0.20765266, -0.46590003], [0.84138155, -0.9276225, -0.32545793, -0.77252215], [-0.23890471, -0.037288006, -0.42536998, -0.24655788]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.16889589, -0.09513443, -0.24702963, -0.84467924], [0.04562974, -0.24083203, -0.2024512, -0.19128312], [0.65530276, -0.7801077, 0.20149614, -0.44066793], [-0.17158562, 0.2859542, -0.014121061, 0.6702811]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.4864707, 0.11541373, 0.3640988, -0.6173036], [0.34746346, 0.030759, -0.37615424, -0.03167375], [0.49353784, -0.27926895, -0.26754513, -0.06590121], [-0.42753345, -0.09161461, -0.3102058, -0.22481634]]], [[[0.007876359, -0.3951988, 0.05565301, -0.11668123], [-0.061648875, -0.3147027, 0.0022256465, -0.4170613], [0.54406863, -0.20265156, -0.1976334, -0.46953517], [-0.1680731, 0.16391398, -0.49734992, 0.13039753]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.11883754, 0.19823699, 0.045379855, 0.32375816], [0.11770367, -0.18533918, -0.04328479, -0.61093134], [0.7527994, -0.31336316, 0.15661053, -0.5106034], [-0.29349425, 0.1376311, -0.051912528, 0.026611617]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
left_dangle_count_mat: [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.12767829, 0.04606526, -0.021292783, 0.0062827007]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.06651404, 0.062189706, 0.103618525, -0.16166256], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [-0.1840861, 0.028572999, 0.13675922, -0.16105959], [0.0, 0.0, 0.0, 0.0], [-0.06430323, -0.041394174, 0.02764992, -0.040736895]], [[-0.031084942, -0.0003068878, -0.11656211, -0.013306196], [0.0, 0.0, 0.0, 0.0], [-0.07969477, 0.0019136921, 0.101845406, -0.09543231], [0.0, 0.0, 0.0, 0.0]]],
right_dangle_count_mat: [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.027853534, -0.08416719, -0.06914797, -0.019535165]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.1949238, -0.052536618, -0.05562941, -0.22186516], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.01041594, 0.009914496, -0.007980691, -0.25898892], [0.0, 0.0, 0.0, 0.0], [-0.04572237, -0.07370227, 0.017241055, -0.05643247]], [[-0.16513695, 0.068278275, -0.07809133, -0.05610871], [0.0, 0.0, 0.0, 0.0], [0.038630545, -0.005591149, -0.039018743, -0.08712732], [0.0, 0.0, 0.0, 0.0]]],
helix_end_count_mat: [[0.0, 0.0, 0.0, -0.9482646], [0.0, 0.0, -0.4899853, 0.0], [0.0, -0.8379484, 0.0, -1.0275754], [-0.9081059, 0.0, -0.3815579, 0.0]],
base_pair_count_mat: [[0.0, 0.0, 0.0, 0.4908563], [0.0, 0.0, 1.3916214, 0.0], [0.0, 1.3916214, 0.0, -0.01904883], [0.4908563, 0.0, -0.01904883, 0.0]],
interior_loop_length_count_mat_explicit: [[-0.13355005, 0.027280921, -0.1714551, -0.22966616], [0.027280921, -0.10597898, -0.07002166, 0.27826035], [-0.1714551, -0.07002166, -0.024328109, 0.3106259], [-0.22966616, 0.27826035, 0.3106259, -0.3214808]],
bulge_loop_0x1_length_counts: [-0.12194108, -0.067679696, 0.015362542, -0.003332755],
interior_loop_1x1_length_count_mat: [[0.29149178, 0.0982638, -0.361565, -0.19486451], [0.0982638, -0.1588187, 0.43019202, 0.13756652], [-0.361565, 0.43019202, -0.11816626, -0.41341743], [-0.19486451, 0.13756652, -0.41341743, 0.14547239]],
multi_loop_base_count: -1.2084908,
multi_loop_basepairing_count: -0.93465674,
multi_loop_accessible_baseunpairing_count: -0.2410311,
external_loop_accessible_basepairing_count: -0.011629753,
external_loop_accessible_baseunpairing_count: -0.2675455,
match_2_match_count: 3.6943681,
match_2_insert_count: 0.22768304,
insert_extend_count: 1.04738,
insert_switch_count: -7.3405485,
init_match_count: 0.40612477,
init_insert_count: -0.35089523,
insert_counts: [0.0073923487, -0.06743043, -0.057406835, -0.009120148],
align_count_mat: [[0.5019355, -0.398528, -0.24110143, -0.3070254], [-0.398528, 0.64723015, -0.31356782, -0.13539153], [-0.24110143, -0.31356782, 0.6412122, -0.34335682], [-0.3070254, -0.13539153, -0.34335682, 0.44309413]],
hairpin_loop_length_counts_cumulative: [-6.0238533, -9.162835, -8.777505, -6.6054153, -4.721038, -5.346559, -5.468139, -4.9268227, -5.6802506, -5.8813334, -6.2100573, -6.2661667, -7.216544, -7.2703447, -7.3966556, -7.2272415, -7.3386283, -7.1857653, -7.2843175, -7.642531, -7.763297, -8.061177, -8.406787, -8.604227, -8.659652, -8.703307, -8.65925, -8.594047, -8.498428, -8.3320675, -8.099697],
bulge_loop_length_counts_cumulative: [-2.3882532, -3.2805061, -4.1880035, -5.027402, -5.4619746, -6.029617, -5.830065, -5.0756645, -5.679097, -6.399459, -6.9140644, -7.2764745, -7.538819, -7.6990433, -7.7860756, -7.817552, -7.8289256, -7.7992554, -7.7519817, -7.7952538, -7.813383, -7.8915944, -7.9627295, -8.020547, -8.066964, -8.102637, -8.129443, -8.147643, -8.1581745, -8.163335],
interior_loop_length_counts_cumulative: [-0.3785861, -0.72322077, -1.1074613, -1.4309539, -1.7055705, -1.7810093, -1.8492916, -1.8751799, -2.0763714, -2.362882, -2.7289412, -3.0395248, -3.0938938, -3.2346349, -3.2962706, -3.39568, -3.419699, -3.4122245, -3.3961868, -3.4750159, -3.6178534, -3.7684042, -3.842786, -3.9277945, -3.984753, -4.031155, -4.0318503, -4.028942, -4.023261],
interior_loop_length_counts_symm_cumulative: [-0.4807466, -0.8421565, -1.1013664, -1.3404732, -1.2015872, -1.8618493, -2.1670055, -2.2001448, -2.5542798, -2.77233, -2.8966613, -3.05298, -3.1391914, -3.18555, -3.2080052],
interior_loop_length_counts_asymm_cumulative: [-2.120977, -2.6793468, -3.2605138, -3.8776357, -4.186614, -4.3050313, -4.517591, -4.833727, -5.1504316, -5.2421145, -5.4632254, -5.604759, -5.821772, -5.994964, -6.1514554, -6.2558656, -6.3258047, -6.367107, -6.3830037, -6.3693066, -6.3280153, -6.2921004, -6.2638865, -6.2473435, -6.221601, -6.187823, -6.1481075, -6.1506524],
}
}
}