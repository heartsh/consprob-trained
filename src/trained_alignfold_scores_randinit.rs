use AlignfoldScores;
impl AlignfoldScores {
  pub fn load_trained_scores_randinit() -> AlignfoldScores {
    AlignfoldScores {
      hairpin_scores_len: [
        -0.17038672,
        -0.08978434,
        -0.22749598,
        -0.053279728,
        -0.09229403,
        -0.2131781,
        -0.10400811,
        -0.13488802,
        -0.12844126,
        0.01689423,
        -0.036362164,
        -0.07386031,
        -0.13402607,
        -0.024553452,
        -0.05659941,
        0.024291795,
        -0.11053377,
        0.037777763,
        0.015259767,
        0.02731881,
        -0.026010642,
        -0.15178338,
        -0.004687371,
        0.03173151,
        0.071924165,
        0.029195704,
        0.04933845,
        0.00431233,
        0.06863081,
        0.079174556,
        0.0013178622,
      ],
      bulge_scores_len: [
        -0.16155155,
        -0.1419452,
        -0.032664668,
        -0.047550865,
        -0.06125155,
        -0.012460632,
        -0.0051714564,
        -0.123470165,
        0.054836717,
        -0.11322144,
        -0.010879378,
        -0.09387096,
        -0.029392771,
        -0.023797777,
        0.02722013,
        -0.028341876,
        -0.03827947,
        0.0033078962,
        0.004878658,
        -0.065574445,
        0.029346744,
        -0.013304725,
        -0.15478078,
        -0.065891355,
        0.0573388,
        -0.0010711044,
        -0.020780865,
        -0.07518957,
        0.023624597,
        -0.04444341,
      ],
      interior_scores_len: [
        -0.3938649,
        -0.2636343,
        -0.21816483,
        -0.14396726,
        -0.23356943,
        -0.14421584,
        -0.10040272,
        -0.1314681,
        -0.12261006,
        -0.07878574,
        -0.10705559,
        -0.027609685,
        -0.10467364,
        -0.00079898053,
        0.03477994,
        -0.08927569,
        -0.09942474,
        -0.07733417,
        0.026240082,
        -0.0974615,
        -0.022749413,
        0.003871583,
        -0.13703103,
        0.048214246,
        0.05967782,
        -0.0026886477,
        -0.06451455,
        0.03708244,
        -0.050161958,
      ],
      interior_scores_symmetric: [
        -0.21911404,
        -0.154782,
        -0.113224775,
        -0.047460794,
        -0.022145243,
        0.056273546,
        0.040806428,
        -0.034967866,
        0.03196787,
        -0.040421527,
        -0.037833795,
        0.016157256,
        0.013789974,
        0.0013455334,
        0.088186584,
      ],
      interior_scores_asymmetric: [
        -0.17799579,
        -0.10261956,
        -0.02653005,
        0.012293229,
        -0.07797855,
        -0.012742548,
        0.066917315,
        0.090641536,
        -0.01894122,
        0.04009917,
        0.0968505,
        0.03411476,
        -0.04540619,
        0.03905323,
        0.07628209,
        -0.034190457,
        -0.01834368,
        0.06663108,
        0.019576669,
        0.010857546,
        0.07010982,
        0.076644026,
        0.03564907,
        0.09287811,
        -0.009939474,
        0.05825522,
        0.023707312,
        -0.057410024,
      ],
      stack_scores: [
        [
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.07345863],
            [0.0, 0.0, 0.13149695, 0.0],
            [0.0, 0.0997237, 0.0, 0.064326346],
            [0.07009261, 0.0, -0.0038187855, 0.0],
          ],
        ],
        [
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.10776031],
            [0.0, 0.0, 0.13858129, 0.0],
            [0.0, 0.056085296, 0.0, 0.008915581],
            [0.0997237, 0.0, 0.02503165, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
        ],
        [
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.18811144],
            [0.0, 0.0, 0.12403306, 0.0],
            [0.0, 0.13858129, 0.0, 0.072244085],
            [0.13149695, 0.0, 0.08988738, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.052050076],
            [0.0, 0.0, 0.08988738, 0.0],
            [0.0, 0.02503165, 0.0, -0.037542336],
            [-0.0038187855, 0.0, -0.04064924, 0.0],
          ],
        ],
        [
          [
            [0.0, 0.0, 0.0, 0.11137545],
            [0.0, 0.0, 0.18811144, 0.0],
            [0.0, 0.10776031, 0.0, 0.12104153],
            [0.07345863, 0.0, 0.052050076, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.12104153],
            [0.0, 0.0, 0.072244085, 0.0],
            [0.0, 0.008915581, 0.0, -0.0012408096],
            [0.064326346, 0.0, -0.037542336, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
        ],
      ],
      terminal_mismatch_scores: [
        [
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [0.010569408, 0.026298108, 0.004207017, -0.1224698],
            [0.027538776, -0.022981383, -0.004092284, 0.09129637],
            [0.028386554, 0.031604808, 0.08960763, -0.058819108],
            [-0.086311735, -0.065868534, 0.03610203, 0.021394782],
          ],
        ],
        [
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [-0.019536396, -0.009636662, 0.16088456, 0.028331786],
            [-0.035567846, -0.04179617, -0.011479735, 0.090760686],
            [-0.093660496, -0.05445609, -0.02782885, -0.078515485],
            [-0.036301214, -0.030997168, -0.0644976, 0.03022146],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
        ],
        [
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [0.0046151495, -0.030554194, 0.06899743, -0.08113047],
            [0.022237299, 0.035520352, -0.06218253, 0.06810536],
            [-0.089982286, -0.04850514, 0.024904363, 0.07634648],
            [-0.061219294, 0.0418268, -0.13936985, 0.05213461],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [-0.056030698, 0.026822388, -0.02768845, -0.099375084],
            [-0.038630027, -0.06390096, -0.007520002, -0.002267645],
            [-0.11652247, 0.060476173, 0.02173968, 0.050187968],
            [-0.039832637, 0.04493671, -0.040086232, -0.023198675],
          ],
        ],
        [
          [
            [-0.058158956, -0.008021043, -0.078158185, -0.0055594826],
            [0.001208577, 0.10328489, -0.058579374, -0.083464645],
            [0.025702482, -0.033289056, -0.10661825, -0.019474592],
            [-0.03414524, -0.057815738, 0.013603378, -0.0016301078],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
          [
            [-0.014275106, 0.0089486325, 0.082487114, 0.017635697],
            [-0.044323612, 0.030216187, -0.038409643, 0.023679787],
            [0.03813258, -0.08273961, -0.028731827, -0.049480803],
            [-0.044444926, -0.06144817, 0.033034876, -0.019130789],
          ],
          [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
          ],
        ],
      ],
      dangling_scores_left: [
        [
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [-0.08060142, -0.0664269, -0.06162212, -0.040076252],
        ],
        [
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.04718994, -0.030719813, 0.100130804, 0.064777486],
          [0.0, 0.0, 0.0, 0.0],
        ],
        [
          [0.0, 0.0, 0.0, 0.0],
          [0.051588338, -0.06359935, -0.029165246, -0.00024232574],
          [0.0, 0.0, 0.0, 0.0],
          [0.0012823405, -0.04290027, 0.066706665, 0.0668661],
        ],
        [
          [-0.0368893, -0.07561369, 0.0282873, -0.0538544],
          [0.0, 0.0, 0.0, 0.0],
          [0.01168434, -0.055642873, -0.0533748, 0.00790699],
          [0.0, 0.0, 0.0, 0.0],
        ],
      ],
      dangling_scores_right: [
        [
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [-0.009659283, -0.007995983, 0.010416601, -0.033149272],
        ],
        [
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0028627738, -0.049633093, -0.10079737, -0.05822273],
          [0.0, 0.0, 0.0, 0.0],
        ],
        [
          [0.0, 0.0, 0.0, 0.0],
          [-0.014186587, -0.008312292, -0.029743725, 0.015035984],
          [0.0, 0.0, 0.0, 0.0],
          [0.010995314, -0.11489604, -0.05858877, 0.024753207],
        ],
        [
          [0.036849096, 0.028029563, -0.069218636, 0.0044939355],
          [0.0, 0.0, 0.0, 0.0],
          [0.023259128, 0.0039539947, -0.060793307, 0.056813456],
          [0.0, 0.0, 0.0, 0.0],
        ],
      ],
      helix_close_scores: [
        [0.0, 0.0, 0.0, -0.3419855],
        [0.0, 0.0, -0.4235674, 0.0],
        [0.0, -0.4443436, 0.0, -0.17849037],
        [-0.3532036, 0.0, -0.17410484, 0.0],
      ],
      basepair_scores: [
        [0.0, 0.0, 0.0, 0.14305462],
        [0.0, 0.0, 0.48407233, 0.0],
        [0.0, 0.48407233, 0.0, 0.10415505],
        [0.14305462, 0.0, 0.10415505, 0.0],
      ],
      interior_scores_explicit: [
        [-0.046463996, -0.002552162, -0.043161742, -0.037550814],
        [-0.002552162, 0.027662706, 0.027219469, -0.016320983],
        [-0.043161742, 0.027219469, -0.07029779, -0.00019123145],
        [-0.037550814, -0.016320983, -0.00019123145, -0.004242975],
      ],
      bulge_scores_0x1: [-0.06625381, -0.057600886, 0.0349276, -0.06667521],
      interior_scores_1x1: [
        [-0.04863922, 0.08576551, 0.0024484291, -0.06576486],
        [0.08576551, 0.01061286, 0.020605411, 0.02255028],
        [0.0024484291, 0.020605411, 0.04525129, -0.08716582],
        [-0.06576486, 0.02255028, -0.08716582, 0.03547169],
      ],
      multibranch_score_base: -0.03793335,
      multibranch_score_basepair: -0.11361839,
      multibranch_score_unpair: -0.1202405,
      external_score_basepair: -0.088542074,
      external_score_unpair: -0.08195502,
      match2match_score: 2.1686842,
      match2insert_score: -0.70919317,
      insert_extend_score: 0.06548581,
      init_match_score: 0.056625023,
      init_insert_score: -0.074871175,
      insert_scores: [-0.09719215, -0.04565127, -0.03258579, -0.15091705],
      match_scores: [
        [0.10065969, -0.108024, -0.09315697, -0.07431668],
        [-0.108024, 0.15337953, -0.056999374, -0.05123694],
        [-0.09315697, -0.056999374, 0.11941501, -0.12876296],
        [-0.07431668, -0.05123694, -0.12876296, 0.15900229],
      ],
      hairpin_scores_len_cumulative: [
        -0.17038672,
        -0.26017106,
        -0.48766702,
        -0.5409467,
        -0.63324076,
        -0.84641886,
        -0.95042694,
        -1.085315,
        -1.2137562,
        -1.196862,
        -1.2332242,
        -1.3070844,
        -1.4411105,
        -1.4656639,
        -1.5222633,
        -1.4979715,
        -1.6085052,
        -1.5707275,
        -1.5554677,
        -1.5281489,
        -1.5541595,
        -1.7059429,
        -1.7106303,
        -1.6788988,
        -1.6069746,
        -1.5777789,
        -1.5284405,
        -1.5241282,
        -1.4554974,
        -1.3763229,
        -1.375005,
      ],
      bulge_scores_len_cumulative: [
        -0.16155155,
        -0.30349675,
        -0.3361614,
        -0.38371226,
        -0.4449638,
        -0.45742443,
        -0.46259588,
        -0.58606607,
        -0.5312294,
        -0.6444508,
        -0.6553302,
        -0.7492011,
        -0.7785939,
        -0.80239165,
        -0.7751715,
        -0.8035134,
        -0.8417929,
        -0.838485,
        -0.83360636,
        -0.8991808,
        -0.86983407,
        -0.8831388,
        -1.0379195,
        -1.1038109,
        -1.0464721,
        -1.0475432,
        -1.0683241,
        -1.1435137,
        -1.1198891,
        -1.1643325,
      ],
      interior_scores_len_cumulative: [
        -0.3938649, -0.6574992, -0.875664, -1.0196313, -1.2532007, -1.3974165, -1.4978192,
        -1.6292872, -1.7518973, -1.8306831, -1.9377387, -1.9653484, -2.070022, -2.070821,
        -2.036041, -2.1253166, -2.2247415, -2.3020756, -2.2758355, -2.373297, -2.3960464,
        -2.3921747, -2.5292058, -2.4809916, -2.4213138, -2.4240024, -2.488517, -2.4514346,
        -2.5015965,
      ],
      interior_scores_symmetric_cumulative: [
        -0.21911404,
        -0.37389603,
        -0.4871208,
        -0.5345816,
        -0.5567269,
        -0.50045335,
        -0.45964694,
        -0.4946148,
        -0.46264693,
        -0.50306845,
        -0.54090226,
        -0.524745,
        -0.51095504,
        -0.5096095,
        -0.42142293,
      ],
      interior_scores_asymmetric_cumulative: [
        -0.17799579,
        -0.28061533,
        -0.3071454,
        -0.29485217,
        -0.37283072,
        -0.38557327,
        -0.31865597,
        -0.22801444,
        -0.24695566,
        -0.20685649,
        -0.11000599,
        -0.07589123,
        -0.12129742,
        -0.08224419,
        -0.005962096,
        -0.040152553,
        -0.058496233,
        0.008134846,
        0.027711514,
        0.03856906,
        0.10867888,
        0.18532291,
        0.22097197,
        0.31385008,
        0.3039106,
        0.36216584,
        0.38587314,
        0.3284631,
      ],
    }
  }
}
