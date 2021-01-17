extern crate rna_algos;
extern crate bio;
extern crate getopts;
extern crate hashbrown;
extern crate itertools;
extern crate num_cpus;
extern crate scoped_threadpool;
extern crate bfgs;
extern crate ndarray;
extern crate rand;

pub mod utils;
pub mod trained_feature_score_sets;

pub use self::scoped_threadpool::Pool;
pub use bio::io::fasta::Reader;
pub use getopts::Options;
pub use hashbrown::HashSet;
pub use itertools::multizip;
pub use std::cmp::Ord;
pub use std::fs::create_dir;
pub use std::fs::File;
pub use std::io::prelude::*;
pub use std::io::BufWriter;
pub use std::marker::{Send, Sync};
pub use std::path::Path;
pub use std::str::from_utf8_unchecked;
pub use utils::*;
pub use rna_algos::centroid_estimator::*;
pub use bio::io::fasta::*;
pub use bio::utils::*;
pub use std::fs::{read_dir, DirEntry};
pub use bfgs::bfgs;
pub use ndarray::prelude::*;
pub use rand::thread_rng;
pub use rand::seq::SliceRandom;
pub use std::f32::INFINITY;
pub use std::io::stdout;

pub type Regularizers = Array1<Regularizer>;
pub type Regularizer = FeatureCount;
pub type BfgsFeatureCounts = Array1<BfgsFeatureCount>;
pub type BfgsFeatureCount = f64;
pub type FeatureCounts = Array1<FeatureCount>;
pub type TrainData<T> = Vec<TrainDatum<T>>;
pub type Prob4dMat<T> = HashMap<PosQuadruple<T>, Prob>;
pub type PartFunc4dMat<T> = HashMap<PosQuadruple<T>, PartFunc>;
pub type TmpPartFuncSetMat<T> = HashMap<PosPair<T>, TmpPartFuncSets>;
pub type FeatureCount = Prob;
pub type TerminalMismatchCount3dMat = [[[FeatureCount; NUM_OF_BASES]; NUM_OF_BASES]; NUM_OF_BASES];
pub type TerminalMismatchCount4dMat = [TerminalMismatchCount3dMat; NUM_OF_BASES];
pub type BasepairAlignCount4dMat = TerminalMismatchCount4dMat;
pub type StackCountMat = TerminalMismatchCount4dMat;
pub type HelixEndCountMat = [[FeatureCount; NUM_OF_BASES]; NUM_OF_BASES];
pub type LoopAlignCountMat = HelixEndCountMat;
pub type HairpinLoopLengthCounts =
  [FeatureCount; CONSPROB_MAX_HAIRPIN_LOOP_LEN - CONSPROB_MIN_HAIRPIN_LOOP_LEN + 1];
pub type BulgeLoopLengthCounts = [FeatureCount; CONSPROB_MAX_TWOLOOP_LEN];
pub type InteriorLoopLengthCounts = [FeatureCount; CONSPROB_MAX_TWOLOOP_LEN - 1];
pub type InteriorLoopLengthCountsSymm = [FeatureCount; CONSPROB_MAX_INTERIOR_LOOP_LEN_SYMM];
pub type InteriorLoopLengthCountsAsymm = [FeatureCount; CONSPROB_MAX_INTERIOR_LOOP_LEN_ASYMM];
pub type DangleCount3dMat = [[[FeatureCount; NUM_OF_BASES]; NUM_OF_BASES]; NUM_OF_BASES];
pub type BasePairCountMat = HelixEndCountMat;
pub type InteriorLoopLengthCountMatExplicit = [[FeatureCount; CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT]; CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT];
pub type BulgeLoop0x1LengthCounts = [FeatureCount; NUM_OF_BASES];
pub type InteriorLoop1x1LengthCountMat = [[FeatureCount; NUM_OF_BASES]; NUM_OF_BASES];
pub type InteriorLoopLengthCountMat =
  [[FeatureCount; CONSPROB_MAX_TWOLOOP_LEN - 1]; CONSPROB_MAX_TWOLOOP_LEN - 1];
#[derive(Clone)]
pub struct TrainDatum<T> {
  pub seq_pair: RealSeqPair,
  pub observed_feature_count_sets: FeatureCountSets,
  pub expected_feature_count_sets: FeatureCountSets,
  pub bpp_mat_pair: SparseProbMatPair<T>,
  pub max_bp_span_pair: (T, T),
  pub part_func: Prob,
}
#[derive(Clone, Debug)]
pub struct FeatureCountSets {
  pub hairpin_loop_length_counts: HairpinLoopLengthCounts,
  pub bulge_loop_length_counts: BulgeLoopLengthCounts,
  pub interior_loop_length_counts: InteriorLoopLengthCounts,
  pub interior_loop_length_counts_symm: InteriorLoopLengthCountsSymm,
  pub interior_loop_length_counts_asymm: InteriorLoopLengthCountsAsymm,
  pub stack_count_mat: StackCountMat,
  pub terminal_mismatch_count_mat: TerminalMismatchCount4dMat,
  pub left_dangle_count_mat: DangleCount3dMat,
  pub right_dangle_count_mat: DangleCount3dMat,
  pub helix_end_count_mat: HelixEndCountMat,
  pub base_pair_count_mat: BasePairCountMat,
  pub interior_loop_length_count_mat_explicit: InteriorLoopLengthCountMatExplicit,
  pub bulge_loop_0x1_length_counts: BulgeLoop0x1LengthCounts,
  pub interior_loop_1x1_length_count_mat: InteriorLoop1x1LengthCountMat,
  pub multi_loop_base_count: FeatureCount,
  pub multi_loop_basepairing_count: FeatureCount,
  pub multi_loop_accessible_baseunpairing_count: FeatureCount,
  pub external_loop_accessible_basepairing_count: FeatureCount,
  pub external_loop_accessible_baseunpairing_count: FeatureCount,
  pub basepair_align_count_mat: BasepairAlignCount4dMat,
  pub loop_align_count_mat: LoopAlignCountMat,
  pub opening_gap_count: FeatureCount,
  pub extending_gap_count: FeatureCount,
  pub hairpin_loop_length_counts_cumulative: HairpinLoopLengthCounts,
  pub bulge_loop_length_counts_cumulative: BulgeLoopLengthCounts,
  pub interior_loop_length_counts_cumulative: InteriorLoopLengthCounts,
  pub interior_loop_length_counts_symm_cumulative: InteriorLoopLengthCountsSymm,
  pub interior_loop_length_counts_asymm_cumulative: InteriorLoopLengthCountsAsymm,
}
#[derive(Clone)]
pub struct TmpPartFuncSets {
  // No basepairings exist.
  pub part_funcs_on_sa: TmpPartFuncs,
  // No basepairings exist on multi-loops.
  pub part_funcs_on_sa_4_ml: TmpPartFuncs,
  // At least two basepairings exist in each multi-loop interval.
  pub part_funcs_4_ml: TmpPartFuncs,
  // Just one basepairing exists in each multi-loop interval.
  pub part_funcs_4_first_bpas_on_mls: TmpPartFuncs,
  // At least one basepairing exists in each multi-loop interval.
  pub part_funcs_4_bpas_on_mls: TmpPartFuncs,
  // Zero or more basepairings exist in each multi-loop interval.
  pub part_funcs_on_mls: TmpPartFuncs,
}
#[derive(Clone)]
pub struct TmpPartFuncs {
  pub part_func_4_align: PartFunc,
  pub part_func_4_insert: PartFunc,
  pub part_func_4_insert_2: PartFunc,
  pub part_func: PartFunc,
}
pub type TmpPartFuncSetMat4El<T> = HashMap<PosPair<T>, TmpPartFuncs>;
#[derive(Clone)]
pub struct StaPartFuncMats<T> {
  pub part_func_4d_mat_4_bpas: PartFunc4dMat<T>,
  pub part_func_4d_mat_4_bpas_accessible_on_els: PartFunc4dMat<T>,
  pub part_func_4d_mat_4_bpas_accessible_on_mls: PartFunc4dMat<T>,
  pub forward_part_func_set_mat_4_external_loop: TmpPartFuncSetMat4El<T>,
  pub backward_part_func_set_mat_4_external_loop: TmpPartFuncSetMat4El<T>,
  pub backward_part_func_set_mat_4_external_loop_decode: TmpPartFuncSetMat4El<T>,
  pub forward_tmp_part_func_set_mats_with_pos_pairs: TmpPartFuncSetMatsWithPosPairs<T>,
  pub backward_tmp_part_func_set_mats_with_pos_pairs: TmpPartFuncSetMatsWithPosPairs<T>,
  pub backward_tmp_part_func_set_mats_with_pos_pairs_decode: TmpPartFuncSetMatsWithPosPairs<T>,
  pub forward_2loop_tmp_part_func_set_mats_with_pos_quadruples: TmpPartFuncSetMatsWithPosQuadruples<T>,
  pub backward_2loop_tmp_part_func_set_mats_with_pos_quadruples: TmpPartFuncSetMatsWithPosQuadruples<T>,
  pub backward_2loop_tmp_part_func_set_mats_with_pos_quadruples_decode: TmpPartFuncSetMatsWithPosQuadruples<T>,
}
pub type RnaId = usize;
pub type RnaIdPair = (RnaId, RnaId);
pub type Prob4dMatsWithRnaIdPairs<T> = HashMap<RnaIdPair, Prob4dMat<T>>;
pub type ProbMats<T> = Vec<SparseProbMat<T>>;
pub type Prob1dMats = Vec<Probs>;
pub type Arg = String;
pub type Args = Vec<Arg>;
pub type FastaId = String;
#[derive(Clone)]
pub struct FastaRecord {
  pub fasta_id: FastaId,
  pub seq: Seq,
}
pub type FastaRecords = Vec<FastaRecord>;
pub type SeqPair<'a> = (SeqSlice<'a>, SeqSlice<'a>);
pub type FreeEnergyPair = (FreeEnergy, FreeEnergy);
pub type SparseFreeEnergyMat<T> = HashMap<PosPair<T>, FreeEnergy>;
pub type PosPairsWithPosPairs<T> = HashMap<PosPair<T>, PosPair<T>>;
pub type BoolsWithPosPairs<T> = HashMap<PosPair<T>, bool>;
pub type ProbMatPair<'a, T> = (&'a SparseProbMat<T>, &'a SparseProbMat<T>);
pub type SsFreeEnergyMatSetPair<'a, T> = (&'a SsFreeEnergyMats<T>, &'a SsFreeEnergyMats<T>);
pub type NumOfThreads = u32;
pub type FreeEnergySetPair<'a> = (&'a FreeEnergies, &'a FreeEnergies);
pub struct StaProbMats<T> {
  pub bpp_mat_pair: SparseProbMatPair<T>,
  pub access_bpp_mat_pair_4_2l: SparseProbMatPair<T>,
  pub access_bpp_mat_pair_4_ml: SparseProbMatPair<T>,
  pub bpp_mat_pair_4_el: SparseProbMatPair<T>,
  pub upp_mat_pair: ProbSetPair,
  pub upp_mat_pair_4_hl: ProbSetPair,
  pub upp_mat_pair_4_2l: ProbSetPair,
  pub upp_mat_pair_4_ml: ProbSetPair,
  pub upp_mat_pair_4_el: ProbSetPair,
}
#[derive(Clone)]
pub struct PctStaProbMats<T> {
  pub max_bpp_mat: SparseProbMat<T>,
  pub bpp_mat: SparseProbMat<T>,
  pub access_bpp_mat_4_2l: SparseProbMat<T>,
  pub access_bpp_mat_4_ml: SparseProbMat<T>,
  pub bpp_mat_4_el: SparseProbMat<T>,
  pub bpp_mat_on_ss: SparseProbMat<T>,
  pub max_upp_mat: Probs,
  pub upp_mat: Probs,
  pub upp_mat_4_hl: Probs,
  pub upp_mat_4_2l: Probs,
  pub upp_mat_4_ml: Probs,
  pub upp_mat_4_el: Probs,
}
pub type SparseProbMatPair<T> = (SparseProbMat<T>, SparseProbMat<T>);
pub type ProbSetPair = (Probs, Probs);
pub type ProbMatSets<T> = Vec<PctStaProbMats<T>>;
pub type StaProbMatsWithRnaIdPairs<T> = HashMap<RnaIdPair, StaProbMats<T>>;
pub type ProbSeqPair<'a> = (&'a Probs, &'a Probs);
pub type Poss<T> = Vec<T>;
pub type TmpPartFuncSetMatsWithPosPairs<T> = HashMap<PosPair<T>, TmpPartFuncSetMat<T>>;
pub type TmpPartFuncSetMatsWithPosQuadruples<T> = HashMap<PosQuadruple<T>, PartFuncSetMat<T>>;
pub type PartFuncSetMat<T> = HashMap<PosPair<T>, TmpPartFuncs>;
pub type RealSeqPair = (Seq, Seq);

impl FeatureCountSets {
  pub fn new(init_val: FeatureCount) -> FeatureCountSets {
    let twod_mat = [[init_val; NUM_OF_BASES]; NUM_OF_BASES];
    let threed_mat = [[[init_val; NUM_OF_BASES]; NUM_OF_BASES]; NUM_OF_BASES];
    let fourd_mat = [[[[init_val; NUM_OF_BASES]; NUM_OF_BASES]; NUM_OF_BASES]; NUM_OF_BASES];
    FeatureCountSets {
      hairpin_loop_length_counts: [init_val; CONSPROB_MAX_HAIRPIN_LOOP_LEN - CONSPROB_MIN_HAIRPIN_LOOP_LEN + 1],
      bulge_loop_length_counts: [init_val; CONSPROB_MAX_TWOLOOP_LEN],
      interior_loop_length_counts: [init_val; CONSPROB_MAX_TWOLOOP_LEN - 1],
      interior_loop_length_counts_symm: [init_val; CONSPROB_MAX_INTERIOR_LOOP_LEN_SYMM],
      interior_loop_length_counts_asymm: [init_val; CONSPROB_MAX_INTERIOR_LOOP_LEN_ASYMM],
      stack_count_mat: fourd_mat,
      terminal_mismatch_count_mat: fourd_mat,
      left_dangle_count_mat: threed_mat,
      right_dangle_count_mat: threed_mat,
      helix_end_count_mat: twod_mat,
      base_pair_count_mat: twod_mat,
      interior_loop_length_count_mat_explicit: [[init_val; CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT]; CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT],
      bulge_loop_0x1_length_counts: [init_val; NUM_OF_BASES],
      interior_loop_1x1_length_count_mat: [[init_val; NUM_OF_BASES]; NUM_OF_BASES],
      multi_loop_base_count: init_val,
      multi_loop_basepairing_count: init_val,
      multi_loop_accessible_baseunpairing_count: init_val,
      external_loop_accessible_basepairing_count: init_val,
      external_loop_accessible_baseunpairing_count: init_val,
      basepair_align_count_mat: fourd_mat,
      loop_align_count_mat: twod_mat,
      opening_gap_count: init_val,
      extending_gap_count: init_val,
      hairpin_loop_length_counts_cumulative: [init_val; CONSPROB_MAX_HAIRPIN_LOOP_LEN - CONSPROB_MIN_HAIRPIN_LOOP_LEN + 1],
      bulge_loop_length_counts_cumulative: [init_val; CONSPROB_MAX_TWOLOOP_LEN],
      interior_loop_length_counts_cumulative: [init_val; CONSPROB_MAX_TWOLOOP_LEN - 1],
      interior_loop_length_counts_symm_cumulative: [init_val; CONSPROB_MAX_INTERIOR_LOOP_LEN_SYMM],
      interior_loop_length_counts_asymm_cumulative: [init_val; CONSPROB_MAX_INTERIOR_LOOP_LEN_ASYMM],
    }
  }

  pub fn get_len(&self) -> usize {
    self.hairpin_loop_length_counts.len()
      + self.bulge_loop_length_counts.len()
      + self.interior_loop_length_counts.len()
      + self.interior_loop_length_counts_symm.len()
      + self.interior_loop_length_counts_asymm.len()
      + self.stack_count_mat.len().pow(4)
      + self.terminal_mismatch_count_mat.len().pow(4)
      + self.left_dangle_count_mat.len().pow(3)
      + self.right_dangle_count_mat.len().pow(3)
      + self.helix_end_count_mat.len().pow(2)
      + self.base_pair_count_mat.len().pow(2)
      + self.interior_loop_length_count_mat_explicit.len().pow(2)
      + self.bulge_loop_0x1_length_counts.len()
      + self.interior_loop_1x1_length_count_mat.len().pow(2)
      + 1
      + 1
      + 1
      + 1
      + 1
      + self.basepair_align_count_mat.len().pow(4)
      + self.loop_align_count_mat.len().pow(2)
      + 1
      + 1
  }

  pub fn get_regularizer_grad(&self, regularizers: &Regularizers) -> Regularizers {
    let mut regularizer_grad = vec![0.; regularizers.len()];
    let mut offset = 0;
    let len = self.hairpin_loop_length_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      let count = self.hairpin_loop_length_counts[i];
      squared_sum += count * count;
    }
    for i in 0 .. len {
      let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.hairpin_loop_length_counts[i]);
      regularizer_grad[offset + i] = regularizer_grad_comp;
    }
    offset += group_size;
    let len = self.bulge_loop_length_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      let count = self.bulge_loop_length_counts[i];
      squared_sum += count * count;
    }
    for i in 0 .. len {
      let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.bulge_loop_length_counts[i]);
      regularizer_grad[offset + i] = regularizer_grad_comp;
    }
    offset += group_size;
    let len = self.interior_loop_length_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      let count = self.interior_loop_length_counts[i];
      squared_sum += count * count;
    }
    for i in 0 .. len {
      let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.interior_loop_length_counts[i]);
      regularizer_grad[offset + i] = regularizer_grad_comp;
    }
    offset += group_size;
    let len = self.interior_loop_length_counts_symm.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      let count = self.interior_loop_length_counts_symm[i];
      squared_sum += count * count;
    }
    for i in 0 .. len {
      let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.interior_loop_length_counts_symm[i]);
      regularizer_grad[offset + i] = regularizer_grad_comp;
    }
    offset += group_size;
    let len = self.interior_loop_length_counts_asymm.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      let count = self.interior_loop_length_counts_asymm[i];
      squared_sum += count * count;
    }
    for i in 0 .. len {
      let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.interior_loop_length_counts_asymm[i]);
      regularizer_grad[offset + i] = regularizer_grad_comp;
    }
    offset += group_size;
    let len = self.stack_count_mat.len();
    let group_size = len.pow(4);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            if !is_canonical(&(k, l)) {continue;}
            let count = self.stack_count_mat[i][j][k][l];
            squared_sum += count * count;
          }
        }
      }
    }
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            if !is_canonical(&(k, l)) {continue;}
            let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.stack_count_mat[i][j][k][l]);
            regularizer_grad[offset + i * len.pow(3) + j * len.pow(2) + k * len + l] = regularizer_grad_comp;
          }
        }
      }
    }
    offset += group_size;
    let len = self.terminal_mismatch_count_mat.len();
    let group_size = len.pow(4);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            let count = self.terminal_mismatch_count_mat[i][j][k][l];
            squared_sum += count * count;
          }
        }
      }
    }
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.terminal_mismatch_count_mat[i][j][k][l]);
            regularizer_grad[offset + i * len.pow(3) + j * len.pow(2) + k * len + l] = regularizer_grad_comp;
          }
        }
      }
    }
    offset += group_size;
    let len = self.left_dangle_count_mat.len();
    let group_size = len.pow(3);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          let count = self.left_dangle_count_mat[i][j][k];
          squared_sum += count * count;
        }
      }
    }
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.left_dangle_count_mat[i][j][k]);
          regularizer_grad[offset + i * len.pow(2) + j * len + k] = regularizer_grad_comp;
        }
      }
    }
    offset += group_size;
    let len = self.right_dangle_count_mat.len();
    let group_size = len.pow(3);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. NUM_OF_BASES {
          let count = self.right_dangle_count_mat[i][j][k];
          squared_sum += count * count;
        }
      }
    }
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.right_dangle_count_mat[i][j][k]);
          regularizer_grad[offset + i * len.pow(2) + j * len + k] = regularizer_grad_comp;
        }
      }
    }
    offset += group_size;
    let len = self.helix_end_count_mat.len();
    let group_size = len.pow(2);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        let count = self.helix_end_count_mat[i][j];
        squared_sum += count * count;
      }
    }
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.helix_end_count_mat[i][j]);
        regularizer_grad[offset + i * len + j] = regularizer_grad_comp;
      }
    }
    offset += group_size;
    let len = self.base_pair_count_mat.len();
    let group_size = len.pow(2);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        let count = self.base_pair_count_mat[i][j];
        squared_sum += count * count;
      }
    }
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.base_pair_count_mat[i][j]);
        regularizer_grad[offset + i * len + j] = regularizer_grad_comp;
      }
    }
    offset += group_size;
    let len = self.interior_loop_length_count_mat_explicit.len();
    let group_size = len.pow(2);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        let count = self.interior_loop_length_count_mat_explicit[i][j];
        squared_sum += count * count;
      }
    }
    for i in 0 .. len {
      for j in 0 .. len {
        let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.interior_loop_length_count_mat_explicit[i][j]);
        regularizer_grad[offset + i * len + j] = regularizer_grad_comp;
      }
    }
    offset += group_size;
    let len = self.bulge_loop_0x1_length_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      let count = self.bulge_loop_0x1_length_counts[i];
      squared_sum += count * count;
    }
    for i in 0 .. len {
      let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.bulge_loop_0x1_length_counts[i]);
      regularizer_grad[offset + i] = regularizer_grad_comp;
    }
    offset += group_size;
    let len = self.interior_loop_1x1_length_count_mat.len();
    let group_size = len.pow(2);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        let count = self.interior_loop_1x1_length_count_mat[i][j];
        squared_sum += count * count;
      }
    }
    for i in 0 .. len {
      for j in 0 .. len {
        let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.interior_loop_1x1_length_count_mat[i][j]);
        regularizer_grad[offset + i * len + j] = regularizer_grad_comp;
      }
    }
    offset += group_size;
    let regularizer_grad_comp = get_regularizer_grad_comp(group_size, self.multi_loop_base_count * self.multi_loop_base_count, self.multi_loop_base_count);
    regularizer_grad[offset] = regularizer_grad_comp;
    offset += 1;
    let regularizer_grad_comp = get_regularizer_grad_comp(group_size, self.multi_loop_basepairing_count * self.multi_loop_basepairing_count, self.multi_loop_basepairing_count);
    regularizer_grad[offset] = regularizer_grad_comp;
    offset += 1;
    let regularizer_grad_comp = get_regularizer_grad_comp(group_size, self.multi_loop_accessible_baseunpairing_count * self.multi_loop_accessible_baseunpairing_count, self.multi_loop_accessible_baseunpairing_count);
    regularizer_grad[offset] = regularizer_grad_comp;
    offset += 1;
    let regularizer_grad_comp = get_regularizer_grad_comp(group_size, self.external_loop_accessible_basepairing_count * self.external_loop_accessible_basepairing_count, self.external_loop_accessible_basepairing_count);
    regularizer_grad[offset] = regularizer_grad_comp;
    offset += 1;
    let regularizer_grad_comp = get_regularizer_grad_comp(group_size, self.external_loop_accessible_baseunpairing_count * self.external_loop_accessible_baseunpairing_count, self.external_loop_accessible_baseunpairing_count);
    regularizer_grad[offset] = regularizer_grad_comp;
    offset += 1;
    let len = self.basepair_align_count_mat.len();
    let group_size = len.pow(4);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            if !is_canonical(&(k, l)) {continue;}
            let count = self.basepair_align_count_mat[i][j][k][l];
            squared_sum += count * count;
          }
        }
      }
    }
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            if !is_canonical(&(k, l)) {continue;}
            let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.basepair_align_count_mat[i][j][k][l]);
            regularizer_grad[offset + i * len.pow(3) + j * len.pow(2) + k * len + l] = regularizer_grad_comp;
          }
        }
      }
    }
    offset += group_size;
    let len = self.loop_align_count_mat.len();
    let group_size = len.pow(2);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        let count = self.loop_align_count_mat[i][j];
        squared_sum += count * count;
      }
    }
    for i in 0 .. len {
      for j in 0 .. len {
        let regularizer_grad_comp = get_regularizer_grad_comp(group_size, squared_sum, self.loop_align_count_mat[i][j]);
        regularizer_grad[offset + i * len + j] = regularizer_grad_comp;
      }
    }
    offset += group_size;
    let regularizer_grad_comp = get_regularizer_grad_comp(group_size, self.opening_gap_count * self.opening_gap_count, self.opening_gap_count);
    regularizer_grad[offset] = regularizer_grad_comp;
    offset += 1;
    let regularizer_grad_comp = get_regularizer_grad_comp(group_size, self.extending_gap_count * self.extending_gap_count, self.extending_gap_count);
    regularizer_grad[offset] = regularizer_grad_comp;
    Array1::from_vec(regularizer_grad)
  }

  pub fn update_regularizers(&self, regularizers: &mut Regularizers) {
    let mut regularizers_tmp = vec![0.; regularizers.len()];
    let mut offset = 0;
    let len = self.hairpin_loop_length_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      let count = self.hairpin_loop_length_counts[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.bulge_loop_length_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      let count = self.bulge_loop_length_counts[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.interior_loop_length_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      let count = self.interior_loop_length_counts[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.interior_loop_length_counts_symm.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      let count = self.interior_loop_length_counts_symm[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.interior_loop_length_counts_asymm.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      let count = self.interior_loop_length_counts_asymm[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.stack_count_mat.len();
    let group_size = len.pow(4);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            if !is_canonical(&(k, l)) {continue;}
            let count = self.stack_count_mat[i][j][k][l];
            squared_sum += count * count;
          }
        }
      }
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            if !is_canonical(&(k, l)) {continue;}
            regularizers_tmp[offset + i * len.pow(3) + j * len.pow(2) + k * len + l] = regularizer;
          }
        }
      }
    }
    offset += group_size;
    let len = self.terminal_mismatch_count_mat.len();
    let group_size = len.pow(4);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            let count = self.terminal_mismatch_count_mat[i][j][k][l];
            squared_sum += count * count;
          }
        }
      }
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            regularizers_tmp[offset + i * len.pow(3) + j * len.pow(2) + k * len + l] = regularizer;
          }
        }
      }
    }
    offset += group_size;
    let len = self.left_dangle_count_mat.len();
    let group_size = len.pow(3);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          let count = self.left_dangle_count_mat[i][j][k];
          squared_sum += count * count;
        }
      }
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          regularizers_tmp[offset + i * len.pow(2) + j * len + k] = regularizer;
        }
      }
    }
    offset += group_size;
    let len = self.right_dangle_count_mat.len();
    let group_size = len.pow(3);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. NUM_OF_BASES {
          let count = self.right_dangle_count_mat[i][j][k];
          squared_sum += count * count;
        }
      }
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          regularizers_tmp[offset + i * len.pow(2) + j * len + k] = regularizer;
        }
      }
    }
    offset += group_size;
    let len = self.helix_end_count_mat.len();
    let group_size = len.pow(2);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        let count = self.helix_end_count_mat[i][j];
        squared_sum += count * count;
      }
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        regularizers_tmp[offset + i * len + j] = regularizer;
      }
    }
    offset += group_size;
    let len = self.base_pair_count_mat.len();
    let group_size = len.pow(2);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        let count = self.base_pair_count_mat[i][j];
        squared_sum += count * count;
      }
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        regularizers_tmp[offset + i * len + j] = regularizer;
      }
    }
    offset += group_size;
    let len = self.interior_loop_length_count_mat_explicit.len();
    let group_size = len.pow(2);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        let count = self.interior_loop_length_count_mat_explicit[i][j];
        squared_sum += count * count;
      }
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      for j in 0 .. len {
        regularizers_tmp[offset + i * len + j] = regularizer;
      }
    }
    offset += group_size;
    let len = self.bulge_loop_0x1_length_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      let count = self.bulge_loop_0x1_length_counts[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.interior_loop_1x1_length_count_mat.len();
    let group_size = len.pow(2);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        let count = self.interior_loop_1x1_length_count_mat[i][j];
        squared_sum += count * count;
      }
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      for j in 0 .. len {
        regularizers_tmp[offset + i * len + j] = regularizer;
      }
    }
    offset += group_size;
    let regularizer = get_regularizer(1, self.multi_loop_base_count * self.multi_loop_base_count);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let regularizer = get_regularizer(1, self.multi_loop_basepairing_count * self.multi_loop_basepairing_count);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let regularizer = get_regularizer(1, self.multi_loop_accessible_baseunpairing_count * self.multi_loop_accessible_baseunpairing_count);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let regularizer = get_regularizer(1, self.external_loop_accessible_basepairing_count * self.external_loop_accessible_basepairing_count);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let regularizer = get_regularizer(1, self.external_loop_accessible_baseunpairing_count * self.external_loop_accessible_baseunpairing_count);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let len = self.basepair_align_count_mat.len();
    let group_size = len.pow(4);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            if !is_canonical(&(k, l)) {continue;}
            let count = self.basepair_align_count_mat[i][j][k][l];
            squared_sum += count * count;
          }
        }
      }
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            if !is_canonical(&(k, l)) {continue;}
            regularizers_tmp[offset + i * len.pow(3) + j * len.pow(2) + k * len + l] = regularizer;
          }
        }
      }
    }
    offset += group_size;
    let len = self.loop_align_count_mat.len();
    let group_size = len.pow(2);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        let count = self.loop_align_count_mat[i][j];
        squared_sum += count * count;
      }
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      for j in 0 .. len {
        regularizers_tmp[offset + i * len + j] = regularizer;
      }
    }
    offset += group_size;
    let regularizer = get_regularizer(1, self.opening_gap_count * self.opening_gap_count);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let regularizer = get_regularizer(1, self.extending_gap_count * self.extending_gap_count);
    regularizers_tmp[offset] = regularizer;
    *regularizers = Array1::from_vec(regularizers_tmp);
  }

  pub fn update<T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord>(&mut self, train_data: &[TrainDatum<T>], regularizers: &mut Regularizers)
  {
    let f = |_: &BfgsFeatureCounts| {
      self.get_log_loss(&train_data[..], regularizers) as BfgsFeatureCount
    };
    let g = |_: &BfgsFeatureCounts| {
      convert_feature_counts_2_bfgs_feature_counts(&self.get_grad(train_data, regularizers))
    };
    match bfgs(convert_feature_counts_2_bfgs_feature_counts(&convert_struct_2_vec(self, false)), f, g) {
      Ok(solution) => {
        *self = convert_vec_2_struct(&convert_bfgs_feature_counts_2_feature_counts(&solution), false);
      }, Err(_) => {
        println!("BFGS failed");
      },
    };
    self.update_regularizers(regularizers);
    self.accumulate();
  }

  pub fn accumulate(&mut self)
  {
    let mut sum = 0.;
    for i in 0 .. self.hairpin_loop_length_counts_cumulative.len() {
      sum += self.hairpin_loop_length_counts[i];
      self.hairpin_loop_length_counts_cumulative[i] = sum;
    }
    let mut sum = 0.;
    for i in 0 .. self.bulge_loop_length_counts_cumulative.len() {
      sum += self.bulge_loop_length_counts[i];
      self.bulge_loop_length_counts_cumulative[i] = sum;
    }
    let mut sum = 0.;
    for i in 0 .. self.interior_loop_length_counts_cumulative.len() {
      sum += self.interior_loop_length_counts[i];
      self.interior_loop_length_counts_cumulative[i] = sum;
    }
    let mut sum = 0.;
    for i in 0 .. self.interior_loop_length_counts_symm_cumulative.len() {
      sum += self.interior_loop_length_counts_symm[i];
      self.interior_loop_length_counts_symm_cumulative[i] = sum;
    }
    let mut sum = 0.;
    for i in 0 .. self.interior_loop_length_counts_asymm_cumulative.len() {
      sum += self.interior_loop_length_counts_asymm[i];
      self.interior_loop_length_counts_asymm_cumulative[i] = sum;
    }
  }

  pub fn get_grad<T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord>(&self, train_data: &[TrainDatum<T>], regularizers: &Regularizers) -> FeatureCounts
  {
    let feature_scores = convert_struct_2_vec(self, false);
    let mut grad = FeatureCountSets::new(0.);
    for train_datum in train_data {
      let ref obs = train_datum.observed_feature_count_sets;
      let ref expect = train_datum.expected_feature_count_sets;
      for i in 0 .. obs.hairpin_loop_length_counts.len() {
        for j in 0 .. i + 1 {
          grad.hairpin_loop_length_counts[j] -= obs.hairpin_loop_length_counts[i] - expect.hairpin_loop_length_counts[i];
        }
      }
      for i in 0 .. obs.bulge_loop_length_counts.len() {
        for j in 0 .. i + 1 {
          grad.bulge_loop_length_counts[j] -= obs.bulge_loop_length_counts[i] - expect.bulge_loop_length_counts[i];
        }
      }
      let len = obs.interior_loop_length_counts.len();
      for i in 0 .. len {
        let obs_count = obs.interior_loop_length_counts[i];
        let expect_count = expect.interior_loop_length_counts[i];
        for j in 0 .. i + 1 {
          grad.interior_loop_length_counts[j] -= obs_count - expect_count;
        }
      }
      let len = obs.interior_loop_length_counts_symm.len();
      for i in 0 .. len {
        let obs_count = obs.interior_loop_length_counts_symm[i];
        let expect_count = expect.interior_loop_length_counts_symm[i];
        for j in 0 .. i + 1 {
          grad.interior_loop_length_counts_symm[j] -= obs_count - expect_count;
        }
      }
      let len = obs.interior_loop_length_counts_asymm.len();
      for i in 0 .. len {
        let obs_count = obs.interior_loop_length_counts_asymm[i];
        let expect_count = expect.interior_loop_length_counts_asymm[i];
        for j in 0 .. i + 1 {
          grad.interior_loop_length_counts_asymm[j] -= obs_count - expect_count;
        }
      }
      for i in 0 .. NUM_OF_BASES {
        for j in 0 .. NUM_OF_BASES {
          if !is_canonical(&(i, j)) {continue;}
          for k in 0 .. NUM_OF_BASES {
            for l in 0 .. NUM_OF_BASES {
              if !is_canonical(&(k, l)) {continue;}
              let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
              let obs_count = obs.stack_count_mat[dict_min_stack.0.0][dict_min_stack.0.1][dict_min_stack.1.0][dict_min_stack.1.1];
              let expect_count = expect.stack_count_mat[dict_min_stack.0.0][dict_min_stack.0.1][dict_min_stack.1.0][dict_min_stack.1.1];
              grad.stack_count_mat[i][j][k][l] -= obs_count - expect_count;
            }
          }
        }
      }
      for i in 0 .. NUM_OF_BASES {
        for j in 0 .. NUM_OF_BASES {
          if !is_canonical(&(i, j)) {continue;}
          for k in 0 .. NUM_OF_BASES {
            for l in 0 .. NUM_OF_BASES {
              let obs_count = obs.terminal_mismatch_count_mat[i][j][k][l];
              let expect_count = expect.terminal_mismatch_count_mat[i][j][k][l];
              grad.terminal_mismatch_count_mat[i][j][k][l] -= obs_count - expect_count;
            }
          }
        }
      }
      for i in 0 .. NUM_OF_BASES {
        for j in 0 .. NUM_OF_BASES {
          if !is_canonical(&(i, j)) {continue;}
          for k in 0 .. NUM_OF_BASES {
            let obs_count = obs.left_dangle_count_mat[i][j][k];
            let expect_count = expect.left_dangle_count_mat[i][j][k];
            grad.left_dangle_count_mat[i][j][k] -= obs_count - expect_count;
          }
        }
      }
      for i in 0 .. NUM_OF_BASES {
        for j in 0 .. NUM_OF_BASES {
          if !is_canonical(&(i, j)) {continue;}
          for k in 0 .. NUM_OF_BASES {
            let obs_count = obs.right_dangle_count_mat[i][j][k];
            let expect_count = expect.right_dangle_count_mat[i][j][k];
            grad.right_dangle_count_mat[i][j][k] -= obs_count - expect_count;
          }
        }
      }
      for i in 0 .. NUM_OF_BASES {
        for j in 0 .. NUM_OF_BASES {
          if !is_canonical(&(i, j)) {continue;}
          let obs_count = obs.helix_end_count_mat[i][j];
          let expect_count = expect.helix_end_count_mat[i][j];
          grad.helix_end_count_mat[i][j] -= obs_count - expect_count;
        }
      }
      for i in 0 .. NUM_OF_BASES {
        for j in 0 .. NUM_OF_BASES {
          if !is_canonical(&(i, j)) {continue;}
          let dict_min_base_pair = get_dict_min_base_pair(&(i, j));
          let obs_count = obs.base_pair_count_mat[dict_min_base_pair.0][dict_min_base_pair.1];
          let expect_count = expect.base_pair_count_mat[dict_min_base_pair.0][dict_min_base_pair.1];
          grad.base_pair_count_mat[i][j] -= obs_count - expect_count;
        }
      }
      let len = obs.interior_loop_length_count_mat_explicit.len();
      for i in 0 .. len {
        for j in 0 .. len {
          let dict_min_loop_len_pair = get_dict_min_loop_len_pair(&(i, j));
          let obs_count = obs.interior_loop_length_count_mat_explicit[dict_min_loop_len_pair.0][dict_min_loop_len_pair.1];
          let expect_count = expect.interior_loop_length_count_mat_explicit[dict_min_loop_len_pair.0][dict_min_loop_len_pair.1];
          grad.interior_loop_length_count_mat_explicit[i][j] -= obs_count - expect_count;
        }
      }
      for i in 0 .. NUM_OF_BASES {
        let obs_count = obs.bulge_loop_0x1_length_counts[i];
        let expect_count = expect.bulge_loop_0x1_length_counts[i];
        grad.bulge_loop_0x1_length_counts[i] -= obs_count - expect_count;
      }
      for i in 0 .. NUM_OF_BASES {
        for j in 0 .. NUM_OF_BASES {
          let dict_min_nuc_pair = get_dict_min_nuc_pair(&(i, j));
          let obs_count = obs.interior_loop_1x1_length_count_mat[dict_min_nuc_pair.0][dict_min_nuc_pair.1];
          let expect_count = expect.interior_loop_1x1_length_count_mat[dict_min_nuc_pair.0][dict_min_nuc_pair.1];
          grad.interior_loop_1x1_length_count_mat[i][j] -= obs_count - expect_count;
        }
      }
      let obs_count = obs.multi_loop_base_count;
      let expect_count = expect.multi_loop_base_count;
      grad.multi_loop_base_count -= obs_count - expect_count;
      let obs_count = obs.multi_loop_basepairing_count;
      let expect_count = expect.multi_loop_basepairing_count;
      grad.multi_loop_basepairing_count -= obs_count - expect_count;
      let obs_count = obs.multi_loop_accessible_baseunpairing_count;
      let expect_count = expect.multi_loop_accessible_baseunpairing_count;
      grad.multi_loop_accessible_baseunpairing_count -= obs_count - expect_count;
      let obs_count = obs.external_loop_accessible_basepairing_count;
      let expect_count = expect.external_loop_accessible_basepairing_count;
      grad.external_loop_accessible_basepairing_count -= obs_count - expect_count;
      let obs_count = obs.external_loop_accessible_baseunpairing_count;
      let expect_count = expect.external_loop_accessible_baseunpairing_count;
      grad.external_loop_accessible_baseunpairing_count -= obs_count - expect_count;
      for i in 0 .. NUM_OF_BASES {
        for j in 0 .. NUM_OF_BASES {
          if !is_canonical(&(i, j)) {continue;}
          for k in 0 .. NUM_OF_BASES {
            for l in 0 .. NUM_OF_BASES {
              if !is_canonical(&(k, l)) {continue;}
              let dict_min_basepair_align = get_dict_min_basepair_align(&(i, j), &(k, l));
              let obs_count = obs.basepair_align_count_mat[dict_min_basepair_align.0.0][dict_min_basepair_align.0.1][dict_min_basepair_align.1.0][dict_min_basepair_align.1.1];
              let expect_count = expect.basepair_align_count_mat[dict_min_basepair_align.0.0][dict_min_basepair_align.0.1][dict_min_basepair_align.1.0][dict_min_basepair_align.1.1];
              grad.basepair_align_count_mat[i][j][k][l] -= obs_count - expect_count;
            }
          }
        }
      }
      for i in 0 .. NUM_OF_BASES {
        for j in 0 .. NUM_OF_BASES {
          let dict_min_loop_align = get_dict_min_loop_align(&(i, j));
          let obs_count = obs.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1];
          let expect_count = expect.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1];
          grad.loop_align_count_mat[i][j] -= obs_count - expect_count;
        }
      }
      let obs_count = obs.opening_gap_count;
      let expect_count = expect.opening_gap_count;
      grad.opening_gap_count -= obs_count - expect_count;
      let obs_count = obs.extending_gap_count;
      let expect_count = expect.extending_gap_count;
      grad.extending_gap_count -= obs_count - expect_count;
    }
    let regularizer_grad = self.get_regularizer_grad(regularizers);
    convert_struct_2_vec(&grad, false) + ((regularizer_grad * feature_scores.clone() + regularizers.clone()) * feature_scores.clone() + regularizers.clone() * feature_scores) / 2.
  }

  pub fn get_log_loss<T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord>(&self, train_data: &[TrainDatum<T>], regularizers: &Regularizers) -> FeatureCount {
    let mut log_likelihood = 0.;
    let feature_scores_cumulative = convert_struct_2_vec(self, true);
    for train_datum in train_data {
      let ref obs = train_datum.observed_feature_count_sets;
      log_likelihood += feature_scores_cumulative.dot(&convert_struct_2_vec(obs, false));
      log_likelihood -= train_datum.part_func;
    }
    let feature_scores = convert_struct_2_vec(self, false);
    let product = regularizers.clone() * feature_scores.clone();
    - log_likelihood + product.dot(&feature_scores) / 2.
  }
}

impl<T: Hash + ToPrimitive + Clone> StaProbMats<T> {
  pub fn origin() -> StaProbMats<T> {
    let prob_mat_pair = (SparseProbMat::<T>::default(), SparseProbMat::<T>::default());
    let prob_set_pair = (Vec::new(), Vec::new());
    StaProbMats {
      bpp_mat_pair: prob_mat_pair.clone(),
      access_bpp_mat_pair_4_2l: prob_mat_pair.clone(),
      access_bpp_mat_pair_4_ml: prob_mat_pair.clone(),
      bpp_mat_pair_4_el: prob_mat_pair,
      upp_mat_pair: prob_set_pair.clone(),
      upp_mat_pair_4_hl: prob_set_pair.clone(),
      upp_mat_pair_4_2l: prob_set_pair.clone(),
      upp_mat_pair_4_ml: prob_set_pair.clone(),
      upp_mat_pair_4_el: prob_set_pair,
    }
  }
  pub fn new(seq_len_pair: &PosPair<T>) -> StaProbMats<T> {
    let prob_mat_pair = (SparseProbMat::<T>::default(), SparseProbMat::<T>::default());
    let prob_set_pair = (
      vec![NEG_INFINITY; seq_len_pair.0.to_usize().unwrap()],
      vec![NEG_INFINITY; seq_len_pair.1.to_usize().unwrap()],
    );
    StaProbMats {
      bpp_mat_pair: prob_mat_pair.clone(),
      access_bpp_mat_pair_4_2l: prob_mat_pair.clone(),
      access_bpp_mat_pair_4_ml: prob_mat_pair.clone(),
      bpp_mat_pair_4_el: prob_mat_pair,
      upp_mat_pair: (
        vec![1.; seq_len_pair.0.to_usize().unwrap()],
        vec![1.; seq_len_pair.1.to_usize().unwrap()],
      ),
      upp_mat_pair_4_hl: prob_set_pair.clone(),
      upp_mat_pair_4_2l: prob_set_pair.clone(),
      upp_mat_pair_4_ml: prob_set_pair.clone(),
      upp_mat_pair_4_el: prob_set_pair,
    }
  }
}

impl<T: Hash + Clone> PctStaProbMats<T> {
  pub fn origin() -> PctStaProbMats<T> {
    let prob_mat = SparseProbMat::<T>::default();
    let probs = Vec::new();
    PctStaProbMats {
      max_bpp_mat: prob_mat.clone(),
      bpp_mat: prob_mat.clone(),
      access_bpp_mat_4_2l: prob_mat.clone(),
      access_bpp_mat_4_ml: prob_mat.clone(),
      bpp_mat_4_el: prob_mat.clone(),
      bpp_mat_on_ss: prob_mat,
      max_upp_mat: probs.clone(),
      upp_mat: probs.clone(),
      upp_mat_4_hl: probs.clone(),
      upp_mat_4_2l: probs.clone(),
      upp_mat_4_ml: probs.clone(),
      upp_mat_4_el: probs,
    }
  }
  pub fn new(seq_len: usize) -> PctStaProbMats<T> {
    let prob_mat = SparseProbMat::<T>::default();
    let probs = vec![0.; seq_len];
    PctStaProbMats {
      max_bpp_mat: prob_mat.clone(),
      bpp_mat: prob_mat.clone(),
      access_bpp_mat_4_2l: prob_mat.clone(),
      access_bpp_mat_4_ml: prob_mat.clone(),
      bpp_mat_4_el: prob_mat.clone(),
      bpp_mat_on_ss: prob_mat,
      max_upp_mat: probs.clone(),
      upp_mat: probs.clone(),
      upp_mat_4_hl: probs.clone(),
      upp_mat_4_2l: probs.clone(),
      upp_mat_4_ml: probs.clone(),
      upp_mat_4_el: probs,
    }
  }
}

impl TmpPartFuncSets {
  pub fn new() -> TmpPartFuncSets {
    let part_funcs = TmpPartFuncs::new();
    TmpPartFuncSets {
      part_funcs_on_sa: part_funcs.clone(),
      part_funcs_on_sa_4_ml: part_funcs.clone(),
      part_funcs_4_ml: part_funcs.clone(),
      part_funcs_4_first_bpas_on_mls: part_funcs.clone(),
      part_funcs_4_bpas_on_mls: part_funcs.clone(),
      part_funcs_on_mls: part_funcs,
    }
  }
}

impl TmpPartFuncs {
  pub fn new() -> TmpPartFuncs {
    TmpPartFuncs {
      part_func_4_align: NEG_INFINITY,
      part_func_4_insert: NEG_INFINITY,
      part_func_4_insert_2: NEG_INFINITY,
      part_func: NEG_INFINITY,
    }
  }
}

impl<T: Hash + Clone> StaPartFuncMats<T> {
  pub fn new() -> StaPartFuncMats<T> {
    let part_func_4d_mat = PartFunc4dMat::<T>::default();
    let part_func_set_mat = TmpPartFuncSetMat4El::<T>::new();
    let tmp_part_func_set_mats_with_pos_pairs = TmpPartFuncSetMatsWithPosPairs::<T>::default();
    let tmp_part_func_set_mats_with_pos_quadruples = TmpPartFuncSetMatsWithPosQuadruples::<T>::default();
    StaPartFuncMats {
      part_func_4d_mat_4_bpas: part_func_4d_mat.clone(),
      part_func_4d_mat_4_bpas_accessible_on_els: part_func_4d_mat.clone(),
      part_func_4d_mat_4_bpas_accessible_on_mls: part_func_4d_mat,
      forward_part_func_set_mat_4_external_loop: part_func_set_mat.clone(),
      backward_part_func_set_mat_4_external_loop: part_func_set_mat.clone(),
      backward_part_func_set_mat_4_external_loop_decode: part_func_set_mat,
      forward_tmp_part_func_set_mats_with_pos_pairs: tmp_part_func_set_mats_with_pos_pairs.clone(),
      backward_tmp_part_func_set_mats_with_pos_pairs: tmp_part_func_set_mats_with_pos_pairs.clone(),
      backward_tmp_part_func_set_mats_with_pos_pairs_decode: tmp_part_func_set_mats_with_pos_pairs,
      forward_2loop_tmp_part_func_set_mats_with_pos_quadruples: tmp_part_func_set_mats_with_pos_quadruples.clone(),
      backward_2loop_tmp_part_func_set_mats_with_pos_quadruples: tmp_part_func_set_mats_with_pos_quadruples.clone(),
      backward_2loop_tmp_part_func_set_mats_with_pos_quadruples_decode: tmp_part_func_set_mats_with_pos_quadruples,
    }
  }
}

impl FastaRecord {
  pub fn new(fasta_id: FastaId, seq: Seq) -> FastaRecord {
    FastaRecord {
      fasta_id: fasta_id,
      seq: seq,
    }
  }
}

impl<T: Hash + Clone + Unsigned + PrimInt + FromPrimitive + Integer> TrainDatum<T> {
  pub fn origin() -> TrainDatum<T> {
    TrainDatum {
      seq_pair: (Seq::new(), Seq::new()),
      observed_feature_count_sets: FeatureCountSets::new(0.),
      expected_feature_count_sets: FeatureCountSets::new(NEG_INFINITY),
      bpp_mat_pair: (SparseProbMat::<T>::default(), SparseProbMat::<T>::default()),
      max_bp_span_pair: (T::zero(), T::zero()),
      part_func: NEG_INFINITY,
    }
  }

  pub fn new(input_file_path: &Path, min_bpp: Prob) -> TrainDatum<T> {
    let fasta_file_reader = Reader::from_file(Path::new(input_file_path)).unwrap();
    let fasta_records: Vec<Record> = fasta_file_reader.records().map(|rec| {rec.unwrap()}).collect();
    let cons_second_struct = fasta_records[2].seq();
    let seq_pair = (
      convert_with_gaps(&fasta_records[0].seq()),
      convert_with_gaps(&fasta_records[1].seq()),
      );
    let seq_pair_without_gaps = (
      remove_gaps(&seq_pair.0),
      remove_gaps(&seq_pair.1),
      );
    let bpp_mat_pair = (
      remove_small_bpps_from_bpp_mat::<T>(&mccaskill_algo(&seq_pair_without_gaps.0[..], false), min_bpp),
      remove_small_bpps_from_bpp_mat::<T>(&mccaskill_algo(&seq_pair_without_gaps.1[..], false), min_bpp),
    );
    let max_bp_span_pair = (
      get_max_bp_span::<T>(&bpp_mat_pair.0),
      get_max_bp_span::<T>(&bpp_mat_pair.1),
    );
    let mut train_datum = TrainDatum {
      seq_pair: seq_pair_without_gaps,
      observed_feature_count_sets: FeatureCountSets::new(0.),
      expected_feature_count_sets: FeatureCountSets::new(NEG_INFINITY),
      bpp_mat_pair: bpp_mat_pair.clone(),
      max_bp_span_pair: max_bp_span_pair,
      part_func: NEG_INFINITY,
    };
    train_datum.convert(&seq_pair, cons_second_struct);
    train_datum.seq_pair.0.insert(0, PSEUDO_BASE);
    train_datum.seq_pair.0.push(PSEUDO_BASE);
    train_datum.seq_pair.1.insert(0, PSEUDO_BASE);
    train_datum.seq_pair.1.push(PSEUDO_BASE);
    train_datum
  }

  pub fn convert(&mut self, seq_pair: &RealSeqPair, dot_bracket_notation: TextSlice) {
    let align_len = dot_bracket_notation.len();
    let mut is_inserting = false;
    let mut is_inserting_2 = is_inserting;
    for i in 0 .. align_len {
      let char_pair = (seq_pair.0[i], seq_pair.1[i]);
      if dot_bracket_notation[i] != UNPAIRING_BASE {
        is_inserting = false;
        is_inserting_2 = is_inserting;
        continue;
      }
      if char_pair.0 == PSEUDO_BASE && char_pair.1 == PSEUDO_BASE {
        is_inserting = false;
        is_inserting_2 = is_inserting;
      } else if char_pair.1 == PSEUDO_BASE {
        if is_inserting {
          self.observed_feature_count_sets.extending_gap_count += 1.;
        } else {
          self.observed_feature_count_sets.opening_gap_count += 1.;
          is_inserting = true;
          is_inserting_2 = false;
        }
      } else if char_pair.0 == PSEUDO_BASE {
        if is_inserting_2 {
          self.observed_feature_count_sets.extending_gap_count += 1.;
        } else {
          self.observed_feature_count_sets.opening_gap_count += 1.;
          is_inserting_2 = true;
          is_inserting = false;
        }
      } else {
        let dict_min_loop_align = get_dict_min_loop_align(&char_pair);
        self.observed_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1] += 1.;
        is_inserting = false;
        is_inserting_2 = is_inserting;
      }
    }
    let mut stack = Vec::new();
    let mut cons_second_struct = HashSet::<(usize, usize)>::default();
    for i in 0 .. align_len {
      let notation_char = dot_bracket_notation[i];
      if notation_char == BASE_PAIRING_LEFT_BASE {
        stack.push(i);
      } else if notation_char == BASE_PAIRING_RIGHT_BASE {
        let pos = stack.pop().unwrap();
        let base_pair_1 = (seq_pair.0[pos], seq_pair.0[i]);
        if base_pair_1.0 == PSEUDO_BASE || base_pair_1.1 == PSEUDO_BASE {continue;}
        if !is_canonical(&base_pair_1) {continue;}
        let base_pair_2 = (seq_pair.1[pos], seq_pair.1[i]);
        if base_pair_2.0 == PSEUDO_BASE || base_pair_2.1 == PSEUDO_BASE {continue;}
        if !is_canonical(&base_pair_2) {continue;}
        cons_second_struct.insert((pos, i));
        let dict_min_base_pair_1 = get_dict_min_base_pair(&base_pair_1);
        self.observed_feature_count_sets.base_pair_count_mat[dict_min_base_pair_1.0][dict_min_base_pair_1.1] += 1.;
        let dict_min_base_pair_2 = get_dict_min_base_pair(&base_pair_2);
        self.observed_feature_count_sets.base_pair_count_mat[dict_min_base_pair_2.0][dict_min_base_pair_2.1] += 1.;
      }
    }
    let mut loop_struct = HashMap::<(usize, usize), Vec<(usize, usize)>>::default();
    let mut stored_pos_pairs = HashSet::<(usize, usize)>::default();
    for substr_len in 2 .. align_len + 1 {
      for i in 0 .. align_len - substr_len + 1 {
        let mut is_pos_pair_found = false;
        let j = i + substr_len - 1;
        if cons_second_struct.contains(&(i, j)) {
          is_pos_pair_found = true;
          cons_second_struct.remove(&(i, j));
          stored_pos_pairs.insert((i, j));
        }
        if is_pos_pair_found {
          let mut pos_pairs_in_loop = Vec::new();
          for stored_pos_pair in stored_pos_pairs.iter() {
            if i < stored_pos_pair.0 && stored_pos_pair.1 < j {
              pos_pairs_in_loop.push(*stored_pos_pair);
            }
          }
          for pos_pair in pos_pairs_in_loop.iter() {
            stored_pos_pairs.remove(pos_pair);
          }
          pos_pairs_in_loop.sort();
          loop_struct.insert((i, j), pos_pairs_in_loop);
        }
      }
    }
    for (pos_pair_closing_loop, pos_pairs_in_loop) in loop_struct.iter() {
      let num_of_basepairings_in_loop = pos_pairs_in_loop.len();
      let base_pair = (seq_pair.0[pos_pair_closing_loop.0], seq_pair.0[pos_pair_closing_loop.1]);
      let base_pair_2 = (seq_pair.1[pos_pair_closing_loop.0], seq_pair.1[pos_pair_closing_loop.1]);
      let mismatch_pair = get_mismatch_pair(&seq_pair.0[..], &pos_pair_closing_loop, true);
      let mismatch_pair_2 = get_mismatch_pair(&seq_pair.1[..], &pos_pair_closing_loop, true);
      self.observed_feature_count_sets.basepair_align_count_mat[base_pair.0][base_pair.1][base_pair_2.0][base_pair_2.1] += 1.;
      if num_of_basepairings_in_loop == 0 {
        if mismatch_pair.0 != PSEUDO_BASE && mismatch_pair.1 != PSEUDO_BASE {
          self.observed_feature_count_sets.terminal_mismatch_count_mat[base_pair.0][base_pair.1][mismatch_pair.0][mismatch_pair.1] += 1.;
        }
        if mismatch_pair_2.0 != PSEUDO_BASE && mismatch_pair_2.1 != PSEUDO_BASE {
          self.observed_feature_count_sets.terminal_mismatch_count_mat[base_pair_2.0][base_pair_2.1][mismatch_pair_2.0][mismatch_pair_2.1] += 1.;
        }
        let hairpin_loop_length_pair = (
          get_hairpin_loop_length(&seq_pair.0[..], &pos_pair_closing_loop),
          get_hairpin_loop_length(&seq_pair.1[..], &pos_pair_closing_loop),
          );
        if CONSPROB_MIN_HAIRPIN_LOOP_LEN <= hairpin_loop_length_pair.0 && hairpin_loop_length_pair.0 <= CONSPROB_MAX_HAIRPIN_LOOP_LEN {
          self.observed_feature_count_sets.hairpin_loop_length_counts[hairpin_loop_length_pair.0 - CONSPROB_MIN_HAIRPIN_LOOP_LEN] += 1.;
        }
        if CONSPROB_MIN_HAIRPIN_LOOP_LEN <= hairpin_loop_length_pair.1 && hairpin_loop_length_pair.1 <= CONSPROB_MAX_HAIRPIN_LOOP_LEN {
          self.observed_feature_count_sets.hairpin_loop_length_counts[hairpin_loop_length_pair.1 - CONSPROB_MIN_HAIRPIN_LOOP_LEN] += 1.;
        }
        self.observed_feature_count_sets.helix_end_count_mat[base_pair.0][base_pair.1] += 1.;
      } else if num_of_basepairings_in_loop == 1 {
        let ref pos_pair_in_loop = pos_pairs_in_loop[0];
        let base_pair_3 = (seq_pair.0[pos_pair_in_loop.0], seq_pair.0[pos_pair_in_loop.1]);
        let base_pair_4 = (seq_pair.1[pos_pair_in_loop.0], seq_pair.1[pos_pair_in_loop.1]);
        let twoloop_length_pair = get_2loop_length_pair(&seq_pair.0[..], &pos_pair_closing_loop, &pos_pair_in_loop);
        let sum = twoloop_length_pair.0 + twoloop_length_pair.1;
        if sum == 0 {
          self.observed_feature_count_sets.stack_count_mat[base_pair.0][base_pair.1][base_pair_3.0][base_pair_3.1] += 1.;
        } else if twoloop_length_pair.0 == 0 || twoloop_length_pair.1 == 0 {
          if sum <= CONSPROB_MAX_TWOLOOP_LEN {
            self.observed_feature_count_sets.bulge_loop_length_counts[sum - 1] += 1.;
            if sum == 1 {
              let mismatch = if twoloop_length_pair.0 == 0 {mismatch_pair.1} else {mismatch_pair.0};
              self.observed_feature_count_sets.bulge_loop_0x1_length_counts[mismatch] += 1.;
            }
          }
          self.observed_feature_count_sets.helix_end_count_mat[base_pair.0][base_pair.1] += 1.;
          self.observed_feature_count_sets.helix_end_count_mat[base_pair_3.1][base_pair_3.0] += 1.;
        } else {
          if sum <= CONSPROB_MAX_TWOLOOP_LEN {
            self.observed_feature_count_sets.interior_loop_length_counts[sum - 2] += 1.;
            let diff = get_diff(twoloop_length_pair.0, twoloop_length_pair.1);
            if diff == 0 {
              self.observed_feature_count_sets.interior_loop_length_counts_symm[twoloop_length_pair.0 - 1] += 1.;
            } else {
              self.observed_feature_count_sets.interior_loop_length_counts_asymm[diff - 1] += 1.;
            }
            if twoloop_length_pair.0 == 1 && twoloop_length_pair.1 == 1 {
              let dict_min_mismatch_pair = get_dict_min_mismatch_pair(&mismatch_pair);
              self.observed_feature_count_sets.interior_loop_1x1_length_count_mat[dict_min_mismatch_pair.0][dict_min_mismatch_pair.1] += 1.;
            }
            if twoloop_length_pair.0 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT && twoloop_length_pair.1 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT {
              let dict_min_loop_len_pair = get_dict_min_loop_len_pair(&twoloop_length_pair);
              self.observed_feature_count_sets.interior_loop_length_count_mat_explicit[dict_min_loop_len_pair.0 - 1][dict_min_loop_len_pair.1 - 1] += 1.;
            }
          }
          if mismatch_pair.0 != PSEUDO_BASE && mismatch_pair.1 != PSEUDO_BASE {
            self.observed_feature_count_sets.terminal_mismatch_count_mat[base_pair.0][base_pair.1][mismatch_pair.0][mismatch_pair.1] += 1.;
          }
          let mismatch_pair_3 = get_mismatch_pair(&seq_pair.0[..], pos_pair_in_loop, false);
          if mismatch_pair_3.0 != PSEUDO_BASE && mismatch_pair_3.1 != PSEUDO_BASE {
            self.observed_feature_count_sets.terminal_mismatch_count_mat[base_pair_3.1][base_pair_3.0][mismatch_pair_3.1][mismatch_pair_3.0] += 1.;
          }
          self.observed_feature_count_sets.helix_end_count_mat[base_pair.0][base_pair.1] += 1.;
          self.observed_feature_count_sets.helix_end_count_mat[base_pair_3.1][base_pair_3.0] += 1.;
        }
        let twoloop_length_pair_2 = get_2loop_length_pair(&seq_pair.1[..], &pos_pair_closing_loop, &pos_pair_in_loop);
        let sum_2 = twoloop_length_pair_2.0 + twoloop_length_pair_2.1;
        if sum_2 == 0 {
          self.observed_feature_count_sets.stack_count_mat[base_pair_2.0][base_pair_2.1][base_pair_4.0][base_pair_4.1] += 1.;
        } else if twoloop_length_pair_2.0 == 0 || twoloop_length_pair_2.1 == 0 {
          if sum_2 <= CONSPROB_MAX_TWOLOOP_LEN {
            self.observed_feature_count_sets.bulge_loop_length_counts[sum_2 - 1] += 1.;
            if sum_2 == 1 {
              let mismatch_2 = if twoloop_length_pair_2.0 == 0 {mismatch_pair_2.1} else {mismatch_pair_2.0};
              self.observed_feature_count_sets.bulge_loop_0x1_length_counts[mismatch_2] += 1.;
            }
          }
          self.observed_feature_count_sets.helix_end_count_mat[base_pair_2.0][base_pair_2.1] += 1.;
          self.observed_feature_count_sets.helix_end_count_mat[base_pair_4.1][base_pair_4.0] += 1.;
        } else {
          if sum_2 <= CONSPROB_MAX_TWOLOOP_LEN {
            self.observed_feature_count_sets.interior_loop_length_counts[sum_2 - 2] += 1.;
            let diff_2 = get_diff(twoloop_length_pair_2.0, twoloop_length_pair_2.1);
            if diff_2 == 0 {
              self.observed_feature_count_sets.interior_loop_length_counts_symm[twoloop_length_pair_2.0 - 1] += 1.;
            } else {
              self.observed_feature_count_sets.interior_loop_length_counts_asymm[diff_2 - 1] += 1.;
            }
            if twoloop_length_pair_2.0 == 1 && twoloop_length_pair_2.1 == 1 {
              let dict_min_mismatch_pair_2 = get_dict_min_mismatch_pair(&mismatch_pair_2);
              self.observed_feature_count_sets.interior_loop_1x1_length_count_mat[dict_min_mismatch_pair_2.0][dict_min_mismatch_pair_2.1] += 1.;
            }
            if twoloop_length_pair_2.0 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT && twoloop_length_pair_2.1 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT {
              let dict_min_loop_len_pair_2 = get_dict_min_loop_len_pair(&twoloop_length_pair_2);
              self.observed_feature_count_sets.interior_loop_length_count_mat_explicit[dict_min_loop_len_pair_2.0 - 1][dict_min_loop_len_pair_2.1 - 1] += 1.;
            }
          }
          if mismatch_pair_2.0 != PSEUDO_BASE && mismatch_pair_2.1 != PSEUDO_BASE {
            self.observed_feature_count_sets.terminal_mismatch_count_mat[base_pair_2.0][base_pair_2.1][mismatch_pair_2.0][mismatch_pair_2.1] += 1.;
          }
          let mismatch_pair_4 = get_mismatch_pair(&seq_pair.1[..], pos_pair_in_loop, false);
          if mismatch_pair_4.0 != PSEUDO_BASE && mismatch_pair_4.1 != PSEUDO_BASE {
            self.observed_feature_count_sets.terminal_mismatch_count_mat[base_pair_4.1][base_pair_4.0][mismatch_pair_4.1][mismatch_pair_4.0] += 1.;
          }
          self.observed_feature_count_sets.helix_end_count_mat[base_pair_2.0][base_pair_2.1] += 1.;
          self.observed_feature_count_sets.helix_end_count_mat[base_pair_4.1][base_pair_4.0] += 1.;
        }
      } else {
        if mismatch_pair.0 != PSEUDO_BASE && mismatch_pair.1 != PSEUDO_BASE {
          self.observed_feature_count_sets.left_dangle_count_mat[base_pair.0][base_pair.1][mismatch_pair.0] += 1.;
          self.observed_feature_count_sets.right_dangle_count_mat[base_pair.0][base_pair.1][mismatch_pair.1] += 1.;
        }
        if mismatch_pair_2.0 != PSEUDO_BASE && mismatch_pair_2.1 != PSEUDO_BASE {
          self.observed_feature_count_sets.left_dangle_count_mat[base_pair_2.0][base_pair_2.1][mismatch_pair_2.0] += 1.;
          self.observed_feature_count_sets.right_dangle_count_mat[base_pair_2.0][base_pair_2.1][mismatch_pair_2.1] += 1.;
        }
        for pos_pair_in_loop in pos_pairs_in_loop.iter() {
          let base_pair_3 = (seq_pair.0[pos_pair_in_loop.0], seq_pair.0[pos_pair_in_loop.1]);
          let mismatch_pair_3 = get_mismatch_pair(&seq_pair.0[..], pos_pair_in_loop, false);
          if mismatch_pair_3.0 != PSEUDO_BASE && mismatch_pair_3.1 != PSEUDO_BASE {
            self.observed_feature_count_sets.left_dangle_count_mat[base_pair_3.1][base_pair_3.0][mismatch_pair_3.1] += 1.;
            self.observed_feature_count_sets.right_dangle_count_mat[base_pair_3.1][base_pair_3.0][mismatch_pair_3.0] += 1.;
          }
          let base_pair_4 = (seq_pair.1[pos_pair_in_loop.0], seq_pair.1[pos_pair_in_loop.1]);
          let mismatch_pair_4 = get_mismatch_pair(&seq_pair.1[..], pos_pair_in_loop, false);
          if mismatch_pair_4.0 != PSEUDO_BASE && mismatch_pair_4.1 != PSEUDO_BASE {
            self.observed_feature_count_sets.left_dangle_count_mat[base_pair_4.1][base_pair_4.0][mismatch_pair_4.1] += 1.;
            self.observed_feature_count_sets.right_dangle_count_mat[base_pair_4.1][base_pair_4.0][mismatch_pair_4.0] += 1.;
          }
        }
        self.observed_feature_count_sets.multi_loop_base_count += 2.;
        self.observed_feature_count_sets.multi_loop_basepairing_count += 2.;
        self.observed_feature_count_sets.multi_loop_basepairing_count += 2. * num_of_basepairings_in_loop as Prob;
        let num_of_baseunpairing_nucs_in_multi_loop = get_num_of_multiloop_baseunpairing_nucs(pos_pair_closing_loop, pos_pairs_in_loop, &seq_pair.0[..]);
        self.observed_feature_count_sets.multi_loop_accessible_baseunpairing_count += num_of_baseunpairing_nucs_in_multi_loop as Prob;
        let num_of_baseunpairing_nucs_in_multi_loop_2 = get_num_of_multiloop_baseunpairing_nucs(pos_pair_closing_loop, pos_pairs_in_loop, &seq_pair.1[..]);
        self.observed_feature_count_sets.multi_loop_accessible_baseunpairing_count += num_of_baseunpairing_nucs_in_multi_loop_2 as Prob;
        self.observed_feature_count_sets.helix_end_count_mat[base_pair.0][base_pair.1] += 1.;
        self.observed_feature_count_sets.helix_end_count_mat[base_pair_2.0][base_pair_2.1] += 1.;
      }
    }
    self.observed_feature_count_sets.external_loop_accessible_basepairing_count += 2. * stored_pos_pairs.len() as Prob;
    let mut stored_pos_pairs_sorted = stored_pos_pairs.iter().map(|x| *x).collect::<Vec<(usize, usize)>>();
    stored_pos_pairs_sorted.sort();
    let num_of_baseunpairing_nucs_in_external_loop = get_num_of_externalloop_baseunpairing_nucs(&stored_pos_pairs_sorted, &seq_pair.0[..]);
    self.observed_feature_count_sets.external_loop_accessible_baseunpairing_count += num_of_baseunpairing_nucs_in_external_loop as Prob;
    let num_of_baseunpairing_nucs_in_external_loop_2 = get_num_of_externalloop_baseunpairing_nucs(&stored_pos_pairs_sorted, &seq_pair.1[..]);
    self.observed_feature_count_sets.external_loop_accessible_baseunpairing_count += num_of_baseunpairing_nucs_in_external_loop_2 as Prob;
    for pos_pair_in_loop in stored_pos_pairs.iter() {
      let base_pair = (seq_pair.0[pos_pair_in_loop.0], seq_pair.0[pos_pair_in_loop.1]);
      let base_pair_2 = (seq_pair.1[pos_pair_in_loop.0], seq_pair.1[pos_pair_in_loop.1]);
      let mismatch_pair = get_mismatch_pair(&seq_pair.0[..], &pos_pair_in_loop, true);
      let mismatch_pair_2 = get_mismatch_pair(&seq_pair.1[..], &pos_pair_in_loop, true);
      if mismatch_pair.1 != PSEUDO_BASE {
        self.observed_feature_count_sets.left_dangle_count_mat[base_pair.1][base_pair.0][mismatch_pair.1] += 1.;
      }
      if mismatch_pair.0 != PSEUDO_BASE {
        self.observed_feature_count_sets.right_dangle_count_mat[base_pair.1][base_pair.0][mismatch_pair.0] += 1.;
      }
      if mismatch_pair_2.1 != PSEUDO_BASE {
        self.observed_feature_count_sets.left_dangle_count_mat[base_pair_2.1][base_pair_2.0][mismatch_pair_2.1] += 1.;
      }
      if mismatch_pair_2.0 != PSEUDO_BASE {
        self.observed_feature_count_sets.right_dangle_count_mat[base_pair_2.1][base_pair_2.0][mismatch_pair_2.0] += 1.;
      }
      self.observed_feature_count_sets.basepair_align_count_mat[base_pair.0][base_pair.1][base_pair_2.0][base_pair_2.1] += 1.;
      self.observed_feature_count_sets.helix_end_count_mat[base_pair.1][base_pair.0] += 1.;
      self.observed_feature_count_sets.helix_end_count_mat[base_pair_2.1][base_pair_2.0] += 1.;
    }
  }
}

pub const MAX_GAP_NUM_4_IL: usize = 15;
pub const MIN_GAP_NUM_4_IL: usize = 2;
pub const MAX_GAP_NUM_4_IL_TRAIN: usize = MAX_GAP_NUM_4_IL;
pub const MIN_GAP_NUM_4_IL_TRAIN: usize = MIN_GAP_NUM_4_IL;
pub const DEFAULT_MIN_BPP: Prob = 0.04;
pub const DEFAULT_MIN_BPP_4_TRAIN: Prob = 0.005;
pub const DEFAULT_OFFSET_4_MAX_GAP_NUM: usize = 1;
pub const DEFAULT_OFFSET_4_MAX_GAP_NUM_TRAIN: usize = DEFAULT_OFFSET_4_MAX_GAP_NUM;
pub const NUM_OF_BASES: usize = 4;
pub const CONSPROB_MAX_HAIRPIN_LOOP_LEN: usize = 30;
pub const CONSPROB_MAX_TWOLOOP_LEN: usize = CONSPROB_MAX_HAIRPIN_LOOP_LEN;
pub const CONSPROB_MIN_HAIRPIN_LOOP_LEN: usize = 3;
pub const CONSPROB_MIN_HAIRPIN_LOOP_SPAN: usize = CONSPROB_MIN_HAIRPIN_LOOP_LEN + 2;
pub const CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT: usize = 4;
pub const CONSPROB_MAX_INTERIOR_LOOP_LEN_SYMM: usize = CONSPROB_MAX_TWOLOOP_LEN / 2;
pub const CONSPROB_MAX_INTERIOR_LOOP_LEN_ASYMM: usize = CONSPROB_MAX_TWOLOOP_LEN  - 2;
pub const GAMMA_DIST_ALPHA: FeatureCount = 0.;
pub const GAMMA_DIST_BETA: FeatureCount = 1.;
pub const BPP_MAT_FILE_NAME: &'static str = "bpp_mats.dat";
pub const MAX_BPP_MAT_FILE_NAME: &'static str = "max_bpp_mats.dat";
pub const ACCESS_BPP_MAT_ON_2L_FILE_NAME: &'static str = "access_bpp_mats_on_2l.dat";
pub const ACCESS_BPP_MAT_ON_ML_FILE_NAME: &'static str = "access_bpp_mats_on_ml.dat";
pub const BPP_MAT_ON_EL_FILE_NAME: &'static str = "bpp_mats_on_el.dat";
pub const BPP_MAT_ON_SS_FILE_NAME: &'static str = "bpp_mats_on_ss.dat";
pub const UPP_MAT_FILE_NAME: &'static str = "upp_mats.dat";
pub const MAX_UPP_MAT_FILE_NAME: &'static str = "max_upp_mats.dat";
pub const UPP_MAT_ON_HL_FILE_NAME: &'static str = "upp_mats_on_hl.dat";
pub const UPP_MAT_ON_2L_FILE_NAME: &'static str = "upp_mats_on_2l.dat";
pub const UPP_MAT_ON_ML_FILE_NAME: &'static str = "upp_mats_on_ml.dat";
pub const UPP_MAT_ON_EL_FILE_NAME: &'static str = "upp_mats_on_el.dat";
pub const TRAINED_FEATURE_SCORE_SETS_FILE_PATH: &'static str = "./src/trained_feature_score_sets.rs";

pub fn io_algo_4_prob_mats<T>(
  seq_pair: &SeqPair,
  feature_score_sets: &FeatureCountSets,
  max_bp_span_pair: &PosPair<T>,
  max_gap_num: T,
  max_gap_num_4_il: T,
  bpp_mat_pair: &ProbMatPair<T>,
  produces_access_probs: bool,
  trains_score_params: bool,
  expected_feature_count_sets: &mut FeatureCountSets,
) -> (StaProbMats<T>, Prob)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord,
{
  let (sta_part_func_mats, global_part_func) = get_sta_inside_part_func_mats::<T>(
    seq_pair,
    feature_score_sets,
    max_bp_span_pair,
    max_gap_num,
    max_gap_num_4_il,
    bpp_mat_pair,
    false,
    produces_access_probs,
    trains_score_params,
  );
  (get_sta_prob_mats::<T>(
    seq_pair,
    feature_score_sets,
    max_bp_span_pair,
    max_gap_num,
    max_gap_num_4_il,
    &sta_part_func_mats,
    bpp_mat_pair,
    produces_access_probs,
    global_part_func,
    trains_score_params,
    expected_feature_count_sets,
  ), global_part_func)
}

pub fn get_sta_inside_part_func_mats<T>(
  seq_pair: &SeqPair,
  feature_score_sets: &FeatureCountSets,
  max_bp_span_pair: &PosPair<T>,
  max_gap_num: T,
  max_gap_num_4_il: T,
  bpp_mat_pair: &ProbMatPair<T>,
  is_viterbi: bool,
  produces_access_probs: bool,
  trains_score_params: bool,
) -> (StaPartFuncMats<T>, PartFunc)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord,
{
  let seq_len_pair = (
    T::from_usize(seq_pair.0.len()).unwrap(),
    T::from_usize(seq_pair.1.len()).unwrap(),
  );
  let pseudo_pos_quadruple = (
    T::zero(),
    seq_len_pair.0 - T::one(),
    T::zero(),
    seq_len_pair.1 - T::one(),
  );
  let mut sta_part_func_mats = StaPartFuncMats::<T>::new();
  for substr_len_1 in range_inclusive(
    T::from_usize(CONSPROB_MIN_HAIRPIN_LOOP_SPAN).unwrap(),
    max_bp_span_pair.0,
  ) {
    for i in range(T::one(), seq_len_pair.0 - substr_len_1) {
      let j = i + substr_len_1 - T::one();
      let (long_i, long_j) = (i.to_usize().unwrap(), j.to_usize().unwrap());
      let base_pair = (seq_pair.0[long_i], seq_pair.0[long_j]);
      if !is_canonical(&base_pair) {
        continue;
      }
      if !bpp_mat_pair.0.contains_key(&(i, j)) {
        continue;
      }
      let long_pos_pair = (long_i, long_j);
      let multi_loop_closing_basepairing_score = get_consprob_multi_loop_closing_basepairing_score(
        feature_score_sets,
        seq_pair.0,
        &long_pos_pair,
      );
      let multi_loop_accessible_basepairing_score =
        get_consprob_multi_loop_accessible_basepairing_score(
          feature_score_sets,
          seq_pair.0,
          &long_pos_pair,
        );
      let external_loop_accessible_basepairing_score =
        get_consprob_external_loop_accessible_basepairing_score(
          feature_score_sets,
          seq_pair.0,
          &long_pos_pair,
        );
      let hairpin_loop_score =
        get_consprob_hairpin_loop_score(feature_score_sets, seq_pair.0, &long_pos_pair);
      for substr_len_2 in range_inclusive(
        T::from_usize(CONSPROB_MIN_HAIRPIN_LOOP_SPAN).unwrap(),
        max_bp_span_pair.1,
      ) {
        for k in range(T::one(), seq_len_pair.1 - substr_len_2) {
          if !is_min_gap_ok(&(i, k), &pseudo_pos_quadruple, max_gap_num_4_il) {
            continue;
          }
          let l = k + substr_len_2 - T::one();
          if !is_min_gap_ok(&(j, l), &pseudo_pos_quadruple, max_gap_num_4_il) {
            continue;
          }
          let (long_k, long_l) = (k.to_usize().unwrap(), l.to_usize().unwrap());
          let base_pair_2 = (seq_pair.1[long_k], seq_pair.1[long_l]);
          if !is_canonical(&base_pair_2) {
            continue;
          }
          if !bpp_mat_pair.1.contains_key(&(k, l)) {
            continue;
          }
          let pos_quadruple = (i, j, k, l);
          let basepair_align_score = feature_score_sets.basepair_align_count_mat[base_pair.0]
            [base_pair.1][base_pair_2.0][base_pair_2.1];
          let (forward_tmp_part_func_set_mat, part_func_on_sa, part_func_4_ml, forward_tmp_part_func_set_mat_4_2loop, _, _) =
            get_tmp_part_func_set_mat::<T>(
              &seq_pair,
              feature_score_sets,
              max_gap_num_4_il,
              &pos_quadruple,
              &sta_part_func_mats,
              bpp_mat_pair,
              is_viterbi,
              true,
              produces_access_probs,
              trains_score_params,
            );
          let (backward_tmp_part_func_set_mat, _, _, backward_tmp_part_func_set_mat_4_2loop, backward_tmp_part_func_set_mat_4_decode, backward_tmp_part_func_set_mat_4_2loop_decode) = get_tmp_part_func_set_mat::<T>(
            &seq_pair,
            feature_score_sets,
            max_gap_num_4_il,
            &pos_quadruple,
            &sta_part_func_mats,
            bpp_mat_pair,
            is_viterbi,
            false,
            produces_access_probs,
            trains_score_params,
          );
          let mut sum = NEG_INFINITY;
          let long_pos_pair_2 = (long_k, long_l);
          let hairpin_loop_score_2 =
            get_consprob_hairpin_loop_score(feature_score_sets, seq_pair.1, &long_pos_pair_2);
          let score =
            basepair_align_score + hairpin_loop_score + hairpin_loop_score_2 + part_func_on_sa;
          sumormax(&mut sum, score, is_viterbi);
          for m in range(i + T::one(), j) {
            let long_m = m.to_usize().unwrap();
            for n in range(m + T::one(), j) {
              let long_n = n.to_usize().unwrap();
              let base_pair_3 = (seq_pair.0[long_m], seq_pair.0[long_n]);
              if !is_canonical(&base_pair_3) {
                continue;
              }
              if !bpp_mat_pair.0.contains_key(&(m, n)) {
                continue;
              }
              if long_m - long_i - 1 + long_j - long_n - 1 > CONSPROB_MAX_TWOLOOP_LEN {
                continue;
              }
              let long_pos_pair_3 = (long_m, long_n);
              let twoloop_score = get_consprob_twoloop_score(
                feature_score_sets,
                seq_pair.0,
                &long_pos_pair,
                &long_pos_pair_3,
              );
              for o in range(k + T::one(), l) {
                if !is_min_gap_ok(&(m, o), &pseudo_pos_quadruple, max_gap_num_4_il) {
                  continue;
                }
                let long_o = o.to_usize().unwrap();
                for p in range(o + T::one(), l) {
                  if !is_min_gap_ok(&(n, p), &pseudo_pos_quadruple, max_gap_num_4_il) {
                    continue;
                  }
                  let long_p = p.to_usize().unwrap();
                  let base_pair_4 = (seq_pair.1[long_o], seq_pair.1[long_p]);
                  if !is_canonical(&base_pair_4) {
                    continue;
                  }
                  if long_o - long_k - 1 + long_l - long_p - 1 > CONSPROB_MAX_TWOLOOP_LEN {
                    continue;
                  }
                  let pos_quadruple_2 = (m, n, o, p);
                  match sta_part_func_mats
                    .part_func_4d_mat_4_bpas
                    .get(&pos_quadruple_2)
                  {
                    Some(&part_func) => {
                      let forward_term =
                        match forward_tmp_part_func_set_mat.get(&(m - T::one(), o - T::one())) {
                          Some(part_func_sets) => part_func_sets.part_funcs_on_sa.part_func,
                          None => NEG_INFINITY,
                        };
                      let backward_term =
                        match backward_tmp_part_func_set_mat.get(&(n + T::one(), p + T::one())) {
                          Some(part_func_sets) => part_func_sets.part_funcs_on_sa.part_func,
                          None => NEG_INFINITY,
                        };
                      let part_func_4_2l = forward_term + backward_term;
                      let long_pos_pair_4 = (long_o, long_p);
                      let twoloop_score_2 = get_consprob_twoloop_score(
                        feature_score_sets,
                        seq_pair.1,
                        &long_pos_pair_2,
                        &long_pos_pair_4,
                      );
                      let coefficient =
                        basepair_align_score + twoloop_score + twoloop_score_2 + part_func;
                      sumormax(&mut sum, coefficient + part_func_4_2l, is_viterbi);
                    }
                    None => {}
                  }
                }
              }
            }
          }
          let multi_loop_closing_basepairing_score_2 =
            get_consprob_multi_loop_closing_basepairing_score(
              feature_score_sets,
              seq_pair.1,
              &long_pos_pair_2,
            );
          let score = basepair_align_score
            + multi_loop_closing_basepairing_score
            + multi_loop_closing_basepairing_score_2
            + part_func_4_ml;
          sumormax(&mut sum, score, is_viterbi);
          if sum > NEG_INFINITY {
            sta_part_func_mats
              .part_func_4d_mat_4_bpas
              .insert(pos_quadruple, sum);
            let multi_loop_accessible_basepairing_score_2 =
              get_consprob_multi_loop_accessible_basepairing_score(
                feature_score_sets,
                seq_pair.1,
                &long_pos_pair_2,
              );
            let external_loop_accessible_basepairing_score_2 =
              get_consprob_external_loop_accessible_basepairing_score(
                feature_score_sets,
                seq_pair.1,
                &long_pos_pair_2,
              );
            sta_part_func_mats
              .part_func_4d_mat_4_bpas_accessible_on_els
              .insert(
                pos_quadruple,
                sum
                  + external_loop_accessible_basepairing_score
                  + external_loop_accessible_basepairing_score_2,
              );
            sta_part_func_mats
              .part_func_4d_mat_4_bpas_accessible_on_mls
              .insert(
                pos_quadruple,
                sum
                  + multi_loop_accessible_basepairing_score
                  + multi_loop_accessible_basepairing_score_2,
              );
          }
          match sta_part_func_mats
            .forward_tmp_part_func_set_mats_with_pos_pairs
            .get_mut(&(i, k))
          {
            Some(part_func_set_mat) => {
              *part_func_set_mat = forward_tmp_part_func_set_mat;
            }
            None => {
              sta_part_func_mats
                .forward_tmp_part_func_set_mats_with_pos_pairs
                .insert((i, k), forward_tmp_part_func_set_mat);
            }
          }
          match sta_part_func_mats
            .backward_tmp_part_func_set_mats_with_pos_pairs
            .get_mut(&(j, l))
          {
            Some(part_func_set_mat) => {
              *part_func_set_mat = backward_tmp_part_func_set_mat;
            }
            None => {
              sta_part_func_mats
                .backward_tmp_part_func_set_mats_with_pos_pairs
                .insert((j, l), backward_tmp_part_func_set_mat);
            }
          }
          if produces_access_probs || trains_score_params {
            match sta_part_func_mats
              .backward_tmp_part_func_set_mats_with_pos_pairs_decode
              .get_mut(&(j, l))
            {
              Some(part_func_set_mat) => {
                *part_func_set_mat = backward_tmp_part_func_set_mat_4_decode;
              }
              None => {
                sta_part_func_mats
                  .backward_tmp_part_func_set_mats_with_pos_pairs_decode
                  .insert((j, l), backward_tmp_part_func_set_mat_4_decode);
              }
            }
            sta_part_func_mats
              .forward_2loop_tmp_part_func_set_mats_with_pos_quadruples
              .insert(pos_quadruple, forward_tmp_part_func_set_mat_4_2loop);
            sta_part_func_mats
              .backward_2loop_tmp_part_func_set_mats_with_pos_quadruples
              .insert(pos_quadruple, backward_tmp_part_func_set_mat_4_2loop);
            sta_part_func_mats
              .backward_2loop_tmp_part_func_set_mats_with_pos_quadruples_decode
              .insert(pos_quadruple, backward_tmp_part_func_set_mat_4_2loop_decode);
          }
        }
      }
    }
  }
  let leftmost_pos_pair = (T::zero(), T::zero());
  let rightmost_pos_pair = (seq_len_pair.0 - T::one(), seq_len_pair.1 - T::one());
  let mut part_funcs = TmpPartFuncs::new();
  part_funcs.part_func_4_align = 0.;
  part_funcs.part_func = 0.;
  sta_part_func_mats
    .forward_part_func_set_mat_4_external_loop
    .insert(leftmost_pos_pair, part_funcs.clone());
  sta_part_func_mats
    .backward_part_func_set_mat_4_external_loop
    .insert(rightmost_pos_pair, part_funcs);
  for i in range(T::zero(), seq_len_pair.0 - T::one()) {
    let long_i = i.to_usize().unwrap();
    let base = seq_pair.0[long_i];
    for j in range(T::zero(), seq_len_pair.1 - T::one()) {
      let pos_pair = (i, j);
      if pos_pair == (T::zero(), T::zero()) {
        continue;
      }
      if !is_min_gap_ok(&pos_pair, &pseudo_pos_quadruple, max_gap_num) {
        continue;
      }
      let long_j = j.to_usize().unwrap();
      let mut part_funcs = TmpPartFuncs::new();
      let mut sum = NEG_INFINITY;
      let mut final_sum = sum;
      for k in range(T::one(), i) {
        if !bpp_mat_pair.0.contains_key(&(k, i)) {
          continue;
        }
        for l in range(T::one(), j) {
          let pos_pair_2 = (k - T::one(), l - T::one());
          if !is_min_gap_ok(&pos_pair_2, &pseudo_pos_quadruple, max_gap_num) {
            continue;
          }
          let pos_quadruple = (k, i, l, j);
          match sta_part_func_mats
            .part_func_4d_mat_4_bpas_accessible_on_els
            .get(&pos_quadruple)
          {
            Some(&part_func) => {
              match sta_part_func_mats
                .forward_part_func_set_mat_4_external_loop
                .get(&pos_pair_2)
              {
                Some(part_funcs) => {
                  let score = part_funcs.part_func + part_func;
                  sumormax(&mut sum, score, is_viterbi);
                }
                None => {}
              }
            }
            None => {}
          }
        }
      }
      if i > T::zero() && j > T::zero() {
        let loop_align_score = feature_score_sets.loop_align_count_mat[base][seq_pair.1[long_j]] + 2. * feature_score_sets.external_loop_accessible_baseunpairing_count;
        let pos_pair_2 = (i - T::one(), j - T::one());
        match sta_part_func_mats
          .forward_part_func_set_mat_4_external_loop
          .get(&pos_pair_2)
        {
          Some(part_funcs) => {
            let score = part_funcs.part_func + loop_align_score;
            sumormax(&mut sum, score, is_viterbi);
          }
          None => {}
        }
        part_funcs.part_func_4_align = sum;
        sumormax(&mut final_sum, sum, is_viterbi);
      }
      if i > T::zero() {
        sum = NEG_INFINITY;
        let pos_pair_2 = (i - T::one(), j);
        match sta_part_func_mats
          .forward_part_func_set_mat_4_external_loop
          .get(&pos_pair_2)
        {
          Some(part_funcs) => {
            let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
            sumormax(&mut sum, score, is_viterbi);
            let score = part_funcs.part_func_4_insert + feature_score_sets.extending_gap_count;
            sumormax(&mut sum, score, is_viterbi);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.opening_gap_count;
            sumormax(&mut sum, score, is_viterbi);
          }
          None => {}
        }
        sum += feature_score_sets.external_loop_accessible_baseunpairing_count;
        part_funcs.part_func_4_insert = sum;
        sumormax(&mut final_sum, sum, is_viterbi);
      }
      if j > T::zero() {
        sum = NEG_INFINITY;
        let pos_pair_2 = (i, j - T::one());
        match sta_part_func_mats
          .forward_part_func_set_mat_4_external_loop
          .get(&pos_pair_2)
        {
          Some(part_funcs) => {
            let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
            sumormax(&mut sum, score, is_viterbi);
            let score = part_funcs.part_func_4_insert + feature_score_sets.opening_gap_count;
            sumormax(&mut sum, score, is_viterbi);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.extending_gap_count;
            sumormax(&mut sum, score, is_viterbi);
          }
          None => {}
        }
        sum += feature_score_sets.external_loop_accessible_baseunpairing_count;
        part_funcs.part_func_4_insert_2 = sum;
        sumormax(&mut final_sum, sum, is_viterbi);
      }
      part_funcs.part_func = final_sum;
      sta_part_func_mats
        .forward_part_func_set_mat_4_external_loop
        .insert(pos_pair, part_funcs);
    }
  }
  let global_part_func = sta_part_func_mats.forward_part_func_set_mat_4_external_loop[&(
    seq_len_pair.0 - T::from_usize(2).unwrap(),
    seq_len_pair.1 - T::from_usize(2).unwrap(),
  )]
    .part_func;
  for i in range(T::one(), seq_len_pair.0).rev() {
    let long_i = i.to_usize().unwrap();
    let base = seq_pair.0[long_i];
    for j in range(T::one(), seq_len_pair.1).rev() {
      let pos_pair = (i, j);
      if pos_pair == (seq_len_pair.0 - T::one(), seq_len_pair.1 - T::one()) {
        continue;
      }
      if !is_min_gap_ok(&pos_pair, &pseudo_pos_quadruple, max_gap_num) {
        continue;
      }
      let long_j = j.to_usize().unwrap();
      let mut part_funcs = TmpPartFuncs::new();
      let mut sum = NEG_INFINITY;
      let mut final_sum = sum;
      for k in range(i + T::one(), seq_len_pair.0 - T::one()) {
        if !bpp_mat_pair.0.contains_key(&(i, k)) {
          continue;
        }
        for l in range(j + T::one(), seq_len_pair.1 - T::one()) {
          let pos_pair_2 = (k + T::one(), l + T::one());
          if !is_min_gap_ok(&pos_pair_2, &pseudo_pos_quadruple, max_gap_num) {
            continue;
          }
          let pos_quadruple = (i, k, j, l);
          match sta_part_func_mats
            .part_func_4d_mat_4_bpas_accessible_on_els
            .get(&pos_quadruple)
          {
            Some(&part_func) => {
              match sta_part_func_mats
                .backward_part_func_set_mat_4_external_loop
                .get(&pos_pair_2)
              {
                Some(part_funcs) => {
                  let score = part_funcs.part_func + part_func;
                  sumormax(&mut sum, score, is_viterbi);
                }
                None => {}
              }
            }
            None => {}
          }
        }
      }
      if i < seq_len_pair.0 - T::one() && j < seq_len_pair.1 - T::one() {
        let pos_pair_2 = (i + T::one(), j + T::one());
        let loop_align_score = feature_score_sets.loop_align_count_mat[base][seq_pair.1[long_j]] + 2. * feature_score_sets.external_loop_accessible_baseunpairing_count;
        match sta_part_func_mats
          .backward_part_func_set_mat_4_external_loop
          .get(&pos_pair_2)
        {
          Some(part_funcs) => {
            let score = part_funcs.part_func + loop_align_score;
            sumormax(&mut sum, score, is_viterbi);
          }
          None => {}
        }
        part_funcs.part_func_4_align = sum;
        sumormax(&mut final_sum, sum, is_viterbi);
      }
      if i < seq_len_pair.0 - T::one() {
        sum = NEG_INFINITY;
        let pos_pair_2 = (i + T::one(), j);
        match sta_part_func_mats
          .backward_part_func_set_mat_4_external_loop
          .get(&pos_pair_2)
        {
          Some(part_funcs) => {
            let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
            sumormax(&mut sum, score, is_viterbi);
            let score = part_funcs.part_func_4_insert + feature_score_sets.extending_gap_count;
            sumormax(&mut sum, score, is_viterbi);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.opening_gap_count;
            sumormax(&mut sum, score, is_viterbi);
          }
          None => {}
        }
        sum += feature_score_sets.external_loop_accessible_baseunpairing_count;
        part_funcs.part_func_4_insert = sum;
        sumormax(&mut final_sum, sum, is_viterbi);
      }
      if j < seq_len_pair.1 - T::one() {
        sum = NEG_INFINITY;
        let pos_pair_2 = (i, j + T::one());
        match sta_part_func_mats
          .backward_part_func_set_mat_4_external_loop
          .get(&pos_pair_2)
        {
          Some(part_funcs) => {
            let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
            sumormax(&mut sum, score, is_viterbi);
            let score = part_funcs.part_func_4_insert + feature_score_sets.opening_gap_count;
            sumormax(&mut sum, score, is_viterbi);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.extending_gap_count;
            sumormax(&mut sum, score, is_viterbi);
          }
          None => {}
        }
        sum += feature_score_sets.external_loop_accessible_baseunpairing_count;
        part_funcs.part_func_4_insert_2 = sum;
        sumormax(&mut final_sum, sum, is_viterbi);
      }
      part_funcs.part_func = final_sum;
      sta_part_func_mats
        .backward_part_func_set_mat_4_external_loop
        .insert(pos_pair, part_funcs);
    }
  }
  if produces_access_probs || trains_score_params {
    for i in range(T::one(), seq_len_pair.0).rev() {
      for j in range(T::one(), seq_len_pair.1).rev() {
        let pos_pair = (i, j);
        if !is_min_gap_ok(&pos_pair, &pseudo_pos_quadruple, max_gap_num) {
          continue;
        }
        let mut part_funcs = TmpPartFuncs::new();
        let pos_pair_2 = (i + T::one(), j + T::one());
        match sta_part_func_mats
          .backward_part_func_set_mat_4_external_loop.get(&pos_pair_2) {
          Some(part_funcs_2) => {
            part_funcs.part_func_4_align = part_funcs_2.part_func;
            part_funcs.part_func_4_insert = part_funcs_2.part_func_4_align;
            sumormax(
              &mut part_funcs.part_func_4_insert,
              part_funcs_2.part_func_4_insert_2,
              is_viterbi,
            );
            part_funcs.part_func_4_insert_2 = part_funcs_2.part_func_4_align;
            sumormax(
              &mut part_funcs.part_func_4_insert_2,
              part_funcs_2.part_func_4_insert,
              is_viterbi,
            );
          }, None => {},
        }
        match sta_part_func_mats
          .backward_part_func_set_mat_4_external_loop_decode.get(&(i + T::one(), j)) {
          Some(part_funcs_2) => {
            sumormax(
              &mut part_funcs.part_func_4_insert,
              part_funcs_2.part_func_4_insert + feature_score_sets.extending_gap_count + feature_score_sets.external_loop_accessible_baseunpairing_count,
              is_viterbi,
            );
          }, None => {},
        }
        match sta_part_func_mats
          .backward_part_func_set_mat_4_external_loop.get(&(i, j + T::one())) {
          Some(part_funcs_2) => {
            sumormax(
              &mut part_funcs.part_func_4_insert_2,
              part_funcs_2.part_func_4_insert_2 + feature_score_sets.extending_gap_count + feature_score_sets.external_loop_accessible_baseunpairing_count,
              is_viterbi,
            );
          }, None => {},
        }
        sta_part_func_mats
          .backward_part_func_set_mat_4_external_loop_decode
          .insert(pos_pair, part_funcs);
      }
    }
  }
  (sta_part_func_mats, global_part_func)
}

pub fn get_tmp_part_func_set_mat<T>(
  seq_pair: &SeqPair,
  feature_score_sets: &FeatureCountSets,
  max_gap_num_4_il: T,
  pos_quadruple: &PosQuadruple<T>,
  sta_part_func_mats: &StaPartFuncMats<T>,
  bpp_mat_pair: &ProbMatPair<T>,
  is_viterbi: bool,
  is_forward: bool,
  produces_access_probs: bool,
  trains_score_params: bool,
) -> (TmpPartFuncSetMat<T>, PartFunc, PartFunc, PartFuncSetMat<T>, TmpPartFuncSetMat<T>, PartFuncSetMat<T>)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord,
{
  let seq_len_pair = (
    T::from_usize(seq_pair.0.len()).unwrap(),
    T::from_usize(seq_pair.1.len()).unwrap(),
  );
  let pseudo_pos_quadruple = (
    T::zero(),
    seq_len_pair.0 - T::one(),
    T::zero(),
    seq_len_pair.1 - T::one(),
  );
  let mut cache_is_used = false;
  let &(i, j, k, l) = pos_quadruple;
  let leftmost_pos_pair = if is_forward {
    (i, k)
  } else {
    (i + T::one(), k + T::one())
  };
  let rightmost_pos_pair = if is_forward {
    (j - T::one(), l - T::one())
  } else {
    (j, l)
  };
  let (long_i, long_j, long_k, long_l) = (i.to_usize().unwrap(), j.to_usize().unwrap(), k.to_usize().unwrap(), l.to_usize().unwrap(), );
  let tmp_part_func_set_mats_with_pos_pairs = if is_forward {
    &sta_part_func_mats.forward_tmp_part_func_set_mats_with_pos_pairs
  } else {
    &sta_part_func_mats.backward_tmp_part_func_set_mats_with_pos_pairs
  };
  let mut tmp_part_func_set_mat = match tmp_part_func_set_mats_with_pos_pairs.get(&if is_forward {
    leftmost_pos_pair
  } else {
    rightmost_pos_pair
  }) {
    Some(cache) => {
      cache_is_used = true;
      cache.clone()
    }
    None => TmpPartFuncSetMat::<T>::new(),
  };
  let mut tmp_part_func_set_mat_4_decode = if cache_is_used && !is_forward && (produces_access_probs || trains_score_params) {
    sta_part_func_mats.backward_tmp_part_func_set_mats_with_pos_pairs_decode[&rightmost_pos_pair].clone()
  } else {
    TmpPartFuncSetMat::<T>::new()
  };
  let iter: Poss<T> = if is_forward {
    range(i, j).collect()
  } else {
    range_inclusive(i + T::one(), j).rev().collect()
  };
  let iter_2: Poss<T> = if is_forward {
    range(k, l).collect()
  } else {
    range_inclusive(k + T::one(), l).rev().collect()
  };
  let mut tmp_part_func_set_mat_4_2loop = PartFuncSetMat::<T>::default();
  let mut tmp_part_func_set_mat_4_2loop_decode = PartFuncSetMat::<T>::default();
  for &u in iter.iter() {
    let long_u = u.to_usize().unwrap();
    let base = seq_pair.0[long_u];
    for &v in iter_2.iter() {
      let pos_pair = (u, v);
      if !is_min_gap_ok(&pos_pair, &pseudo_pos_quadruple, max_gap_num_4_il) {
        continue;
      }
      let skips = cache_is_used && tmp_part_func_set_mat.contains_key(&pos_pair);
      if skips && !(produces_access_probs || trains_score_params) {
        continue;
      }
      let mut tmp_part_func_sets = TmpPartFuncSets::new();
      let mut tmp_part_funcs_4_2loop = TmpPartFuncs::new();
      if (is_forward && u == i && v == k) || (!is_forward && u == j && v == l) {
        tmp_part_func_sets.part_funcs_on_sa.part_func_4_align = 0.;
        tmp_part_func_sets.part_funcs_on_sa.part_func = 0.;
        tmp_part_func_sets.part_funcs_on_sa_4_ml.part_func_4_align = 0.;
        tmp_part_func_sets.part_funcs_on_sa_4_ml.part_func = 0.;
        tmp_part_func_set_mat.insert(pos_pair, tmp_part_func_sets);
        continue;
      }
      let long_v = v.to_usize().unwrap();
      let mut sum_on_sa = NEG_INFINITY;
      let mut sum_on_sa_4_ml = sum_on_sa;
      let mut sum_4_ml = sum_on_sa;
      let mut sum_4_first_bpas_on_mls = sum_on_sa;
      let mut sum_4_2loop = sum_on_sa;
      let mut tmp_sum = sum_on_sa;
      // For alignments.
      for m in if is_forward {
        range(i + T::one(), u)
      } else {
        range(u + T::one(), j)
      } {
        if !bpp_mat_pair
          .0
          .contains_key(&if is_forward { (m, u) } else { (u, m) })
        {
          continue;
        }
        for n in if is_forward {
          range(k + T::one(), v)
        } else {
          range(v + T::one(), l)
        } {
          let pos_pair_2 = if is_forward {
            (m - T::one(), n - T::one())
          } else {
            (m + T::one(), n + T::one())
          };
          if !is_min_gap_ok(&pos_pair_2, &pseudo_pos_quadruple, max_gap_num_4_il) {
            continue;
          }
          let pos_quadruple_2 = if is_forward {
            (m, u, n, v)
          } else {
            (u, m, v, n)
          };
          match sta_part_func_mats
            .part_func_4d_mat_4_bpas_accessible_on_mls
            .get(&pos_quadruple_2)
          {
            Some(&part_func) => match tmp_part_func_set_mat.get(&pos_pair_2) {
              Some(part_func_sets) => {
                let ref part_funcs = part_func_sets.part_funcs_4_bpas_on_mls;
                if !skips {
                  let score = part_funcs.part_func + part_func;
                  sumormax(&mut sum_4_ml, score, is_viterbi);
                  let ref part_funcs = part_func_sets.part_funcs_on_sa_4_ml;
                  let score = part_funcs.part_func + part_func;
                  sumormax(&mut sum_4_first_bpas_on_mls, score, is_viterbi);
                }
                if produces_access_probs || trains_score_params {
                  if pos_quadruple_2.0 - i - T::one() + j - pos_quadruple_2.1 - T::one() > T::from_usize(CONSPROB_MAX_TWOLOOP_LEN).unwrap() {
                    continue;
                  }
                  if pos_quadruple_2.2 - k - T::one() + l - pos_quadruple_2.3 - T::one() > T::from_usize(CONSPROB_MAX_TWOLOOP_LEN).unwrap() {
                    continue;
                  }
                  let part_func = sta_part_func_mats
                    .part_func_4d_mat_4_bpas[&pos_quadruple_2];
                  let long_pos_quadruple_2 = (pos_quadruple_2.0.to_usize().unwrap(), pos_quadruple_2.1.to_usize().unwrap(), pos_quadruple_2.2.to_usize().unwrap(), pos_quadruple_2.3.to_usize().unwrap(), );
                  let twoloop_score = get_consprob_twoloop_score(
                    feature_score_sets,
                    seq_pair.0,
                    &(long_i, long_j),
                    &(long_pos_quadruple_2.0, long_pos_quadruple_2.1),
                  );
                  let twoloop_score_2 = get_consprob_twoloop_score(
                    feature_score_sets,
                    seq_pair.1,
                    &(long_k, long_l),
                    &(long_pos_quadruple_2.2, long_pos_quadruple_2.3),
                  );
                  let ref part_funcs = part_func_sets.part_funcs_on_sa;
                  let score = part_funcs.part_func + part_func + twoloop_score + twoloop_score_2;
                  sumormax(&mut sum_4_2loop, score, is_viterbi);
                }
              }
              None => {}
            },
            None => {}
          }
        }
      }
      let pos_pair_2 = if is_forward {
        (u - T::one(), v - T::one())
      } else {
        (u + T::one(), v + T::one())
      };
      let loop_align_score = feature_score_sets.loop_align_count_mat[base][seq_pair.1[long_v]];
      match tmp_part_func_set_mat.get(&pos_pair_2) {
        Some(part_func_sets) => {
          if !skips {
            let ref part_funcs = part_func_sets.part_funcs_4_ml;
            let score = part_funcs.part_func + loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
            sumormax(&mut sum_4_ml, score, is_viterbi);
            let ref part_funcs = part_func_sets.part_funcs_4_first_bpas_on_mls;
            let score = part_funcs.part_func + loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
            sumormax(&mut sum_4_first_bpas_on_mls, score, is_viterbi);
            let ref part_funcs = part_func_sets.part_funcs_on_sa_4_ml;
            let score = part_funcs.part_func + loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
            sumormax(&mut sum_on_sa_4_ml, score, is_viterbi);
            let ref part_funcs = part_func_sets.part_funcs_on_sa;
            let score = part_funcs.part_func + loop_align_score;
            sumormax(&mut sum_on_sa, score, is_viterbi);
          }
          if produces_access_probs || trains_score_params {
            match tmp_part_func_set_mat_4_2loop.get(&pos_pair_2) {
              Some(part_funcs) => {
                let score = part_funcs.part_func + loop_align_score;
                sumormax(&mut sum_4_2loop, score, is_viterbi);
              }, None => {},
            }
          }
        }
        None => {}
      }
      if !skips {
        tmp_part_func_sets.part_funcs_4_ml.part_func_4_align = sum_4_ml;
        sumormax(&mut tmp_sum, sum_4_ml, is_viterbi);
        tmp_part_func_sets
          .part_funcs_4_first_bpas_on_mls
          .part_func_4_align = sum_4_first_bpas_on_mls;
        sumormax(&mut tmp_sum, sum_4_first_bpas_on_mls, is_viterbi);
        tmp_part_func_sets
          .part_funcs_4_bpas_on_mls
          .part_func_4_align = tmp_sum;
          tmp_part_func_sets.part_funcs_on_sa_4_ml.part_func_4_align = sum_on_sa_4_ml;
          sumormax(&mut tmp_sum, sum_on_sa_4_ml, is_viterbi);
          tmp_part_func_sets.part_funcs_on_mls.part_func_4_align = tmp_sum;
          tmp_part_func_sets.part_funcs_on_sa.part_func_4_align = sum_on_sa;
      }
      tmp_part_funcs_4_2loop.part_func_4_align = sum_4_2loop;
      // For inserts.
      let mut sum_on_sa = NEG_INFINITY;
      let mut sum_on_sa_4_ml = sum_on_sa;
      let mut sum_4_ml = sum_on_sa;
      let mut sum_4_first_bpas_on_mls = sum_on_sa;
      let mut sum_4_2loop = sum_on_sa;
      let mut tmp_sum = sum_on_sa;
      let pos_pair_2 = if is_forward {
        (u - T::one(), v)
      } else {
        (u + T::one(), v)
      };
      match tmp_part_func_set_mat.get(&pos_pair_2) {
        Some(part_func_sets) => {
          if !skips {
            let ref part_funcs = part_func_sets.part_funcs_4_ml;
            let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_4_ml, score, is_viterbi);
            let score = part_funcs.part_func_4_insert + feature_score_sets.extending_gap_count;
            sumormax(&mut sum_4_ml, score, is_viterbi);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_4_ml, score, is_viterbi);
            let ref part_funcs = part_func_sets.part_funcs_4_first_bpas_on_mls;
            let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_4_first_bpas_on_mls, score, is_viterbi);
            let score = part_funcs.part_func_4_insert + feature_score_sets.extending_gap_count;
            sumormax(&mut sum_4_first_bpas_on_mls, score, is_viterbi);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_4_first_bpas_on_mls, score, is_viterbi);
            let ref part_funcs = part_func_sets.part_funcs_on_sa_4_ml;
            let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_on_sa_4_ml, score, is_viterbi);
            let score = part_funcs.part_func_4_insert + feature_score_sets.extending_gap_count;
            sumormax(&mut sum_on_sa_4_ml, score, is_viterbi);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_on_sa_4_ml, score, is_viterbi);
            let ref part_funcs = part_func_sets.part_funcs_on_sa;
            let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_on_sa, score, is_viterbi);
            let score = part_funcs.part_func_4_insert + feature_score_sets.extending_gap_count;
            sumormax(&mut sum_on_sa, score, is_viterbi);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_on_sa, score, is_viterbi);
          }
          if produces_access_probs || trains_score_params {
            match tmp_part_func_set_mat_4_2loop.get(&pos_pair_2) {
              Some(part_funcs) => {
                let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
                sumormax(&mut sum_4_2loop, score, is_viterbi);
                let score = part_funcs.part_func_4_insert + feature_score_sets.extending_gap_count;
                sumormax(&mut sum_4_2loop, score, is_viterbi);
                let score = part_funcs.part_func_4_insert_2 + feature_score_sets.opening_gap_count;
                sumormax(&mut sum_4_2loop, score, is_viterbi);
              }, None => {},
            }
          }
        }
        None => {}
      }
      if !skips {
        sum_4_ml += feature_score_sets.multi_loop_accessible_baseunpairing_count;
        sum_4_first_bpas_on_mls += feature_score_sets.multi_loop_accessible_baseunpairing_count;
        sum_on_sa_4_ml += feature_score_sets.multi_loop_accessible_baseunpairing_count;
        tmp_part_func_sets.part_funcs_4_ml.part_func_4_insert = sum_4_ml;
        sumormax(&mut tmp_sum, sum_4_ml, is_viterbi);
        tmp_part_func_sets
          .part_funcs_4_first_bpas_on_mls
          .part_func_4_insert = sum_4_first_bpas_on_mls;
        sumormax(&mut tmp_sum, sum_4_first_bpas_on_mls, is_viterbi);
        tmp_part_func_sets
          .part_funcs_4_bpas_on_mls
          .part_func_4_insert = tmp_sum;
        tmp_part_func_sets.part_funcs_on_sa_4_ml.part_func_4_insert = sum_on_sa_4_ml;
        sumormax(&mut tmp_sum, sum_on_sa_4_ml, is_viterbi);
        tmp_part_func_sets.part_funcs_on_mls.part_func_4_insert = tmp_sum;
        tmp_part_func_sets.part_funcs_on_sa.part_func_4_insert = sum_on_sa;
      }
      tmp_part_funcs_4_2loop.part_func_4_insert = sum_4_2loop;
      // For inserts on the other side.
      let mut sum_on_sa = NEG_INFINITY;
      let mut sum_on_sa_4_ml = sum_on_sa;
      let mut sum_4_ml = sum_on_sa;
      let mut sum_4_first_bpas_on_mls = sum_on_sa;
      let mut sum_4_2loop = sum_on_sa;
      let mut tmp_sum = sum_on_sa;
      let pos_pair_2 = if is_forward {
        (u, v - T::one())
      } else {
        (u, v + T::one())
      };
      match tmp_part_func_set_mat.get(&pos_pair_2) {
        Some(part_func_sets) => {
          if !skips {
            let ref part_funcs = part_func_sets.part_funcs_4_ml;
            let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_4_ml, score, is_viterbi);
            let score = part_funcs.part_func_4_insert + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_4_ml, score, is_viterbi);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.extending_gap_count;
            sumormax(&mut sum_4_ml, score, is_viterbi);
            let ref part_funcs = part_func_sets.part_funcs_4_first_bpas_on_mls;
            let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_4_first_bpas_on_mls, score, is_viterbi);
            let score = part_funcs.part_func_4_insert + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_4_first_bpas_on_mls, score, is_viterbi);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.extending_gap_count;
            sumormax(&mut sum_4_first_bpas_on_mls, score, is_viterbi);
            let ref part_funcs = part_func_sets.part_funcs_on_sa_4_ml;
            let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_on_sa_4_ml, score, is_viterbi);
            let score = part_funcs.part_func_4_insert + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_on_sa_4_ml, score, is_viterbi);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.extending_gap_count;
            sumormax(&mut sum_on_sa_4_ml, score, is_viterbi);
            let ref part_funcs = part_func_sets.part_funcs_on_sa;
            let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_on_sa, score, is_viterbi);
            let score = part_funcs.part_func_4_insert + feature_score_sets.opening_gap_count;
            sumormax(&mut sum_on_sa, score, is_viterbi);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.extending_gap_count;
            sumormax(&mut sum_on_sa, score, is_viterbi);
          }
          if produces_access_probs || trains_score_params {
            match tmp_part_func_set_mat_4_2loop.get(&pos_pair_2) {
              Some(part_funcs) => {
                let score = part_funcs.part_func_4_align + feature_score_sets.opening_gap_count;
                sumormax(&mut sum_4_2loop, score, is_viterbi);
                let score = part_funcs.part_func_4_insert + feature_score_sets.opening_gap_count;
                sumormax(&mut sum_4_2loop, score, is_viterbi);
                let score = part_funcs.part_func_4_insert_2 + feature_score_sets.extending_gap_count;
                sumormax(&mut sum_4_2loop, score, is_viterbi);
              }, None => {},
            }
          }
        }
        None => {}
      }
      if !skips {
        sum_4_ml += feature_score_sets.multi_loop_accessible_baseunpairing_count;
        sum_4_first_bpas_on_mls += feature_score_sets.multi_loop_accessible_baseunpairing_count;
        sum_on_sa_4_ml += feature_score_sets.multi_loop_accessible_baseunpairing_count;
        tmp_part_func_sets.part_funcs_4_ml.part_func_4_insert_2 = sum_4_ml;
        sumormax(&mut tmp_sum, sum_4_ml, is_viterbi);
        tmp_part_func_sets
          .part_funcs_4_first_bpas_on_mls
          .part_func_4_insert_2 = sum_4_first_bpas_on_mls;
        sumormax(&mut tmp_sum, sum_4_first_bpas_on_mls, is_viterbi);
        tmp_part_func_sets
          .part_funcs_4_bpas_on_mls
          .part_func_4_insert_2 = tmp_sum;
          tmp_part_func_sets.part_funcs_on_sa_4_ml.part_func_4_insert_2 = sum_on_sa_4_ml;
          sumormax(&mut tmp_sum, sum_on_sa_4_ml, is_viterbi);
          tmp_part_func_sets.part_funcs_on_mls.part_func_4_insert_2 = tmp_sum;
          tmp_part_func_sets.part_funcs_on_sa.part_func_4_insert_2 = sum_on_sa;
      }
      tmp_part_funcs_4_2loop.part_func_4_insert_2 = sum_4_2loop;
      let mut sum_on_sa = NEG_INFINITY;
      let mut sum_on_sa_4_ml = sum_on_sa;
      let mut sum_4_ml = sum_on_sa;
      let mut sum_4_first_bpas_on_mls = sum_on_sa;
      let mut sum_4_bpas_on_mls = sum_on_sa;
      let mut sum_4_2loop = sum_on_sa;
      let mut sum_on_mls = sum_on_sa;
      if !skips {
        sumormax(
          &mut sum_on_sa,
          tmp_part_func_sets.part_funcs_on_sa.part_func_4_align,
          is_viterbi,
        );
        sumormax(
          &mut sum_on_sa,
          tmp_part_func_sets.part_funcs_on_sa.part_func_4_insert,
          is_viterbi,
        );
        sumormax(
          &mut sum_on_sa,
          tmp_part_func_sets.part_funcs_on_sa.part_func_4_insert_2,
          is_viterbi,
        );
        tmp_part_func_sets.part_funcs_on_sa.part_func = sum_on_sa;
        sumormax(
          &mut sum_on_sa_4_ml,
          tmp_part_func_sets.part_funcs_on_sa_4_ml.part_func_4_align,
          is_viterbi,
        );
        sumormax(
          &mut sum_on_sa_4_ml,
          tmp_part_func_sets.part_funcs_on_sa_4_ml.part_func_4_insert,
          is_viterbi,
        );
        sumormax(
          &mut sum_on_sa_4_ml,
          tmp_part_func_sets.part_funcs_on_sa_4_ml.part_func_4_insert_2,
          is_viterbi,
        );
        tmp_part_func_sets.part_funcs_on_sa_4_ml.part_func = sum_on_sa_4_ml;
        sumormax(
          &mut sum_4_ml,
          tmp_part_func_sets.part_funcs_4_ml.part_func_4_align,
          is_viterbi,
        );
        sumormax(
          &mut sum_4_ml,
          tmp_part_func_sets.part_funcs_4_ml.part_func_4_insert,
          is_viterbi,
        );
        sumormax(
          &mut sum_4_ml,
          tmp_part_func_sets.part_funcs_4_ml.part_func_4_insert_2,
          is_viterbi,
        );
        tmp_part_func_sets.part_funcs_4_ml.part_func = sum_4_ml;
        sumormax(
          &mut sum_4_first_bpas_on_mls,
          tmp_part_func_sets
            .part_funcs_4_first_bpas_on_mls
            .part_func_4_align,
          is_viterbi,
        );
        sumormax(
          &mut sum_4_first_bpas_on_mls,
          tmp_part_func_sets
            .part_funcs_4_first_bpas_on_mls
            .part_func_4_insert,
          is_viterbi,
        );
        sumormax(
          &mut sum_4_first_bpas_on_mls,
          tmp_part_func_sets
            .part_funcs_4_first_bpas_on_mls
            .part_func_4_insert_2,
          is_viterbi,
        );
        tmp_part_func_sets.part_funcs_4_first_bpas_on_mls.part_func = sum_4_first_bpas_on_mls;
        sumormax(
          &mut sum_4_bpas_on_mls,
          tmp_part_func_sets
            .part_funcs_4_bpas_on_mls
            .part_func_4_align,
          is_viterbi,
        );
        sumormax(
          &mut sum_4_bpas_on_mls,
          tmp_part_func_sets
            .part_funcs_4_bpas_on_mls
            .part_func_4_insert,
          is_viterbi,
        );
        sumormax(
          &mut sum_4_bpas_on_mls,
          tmp_part_func_sets
            .part_funcs_4_bpas_on_mls
            .part_func_4_insert_2,
          is_viterbi,
        );
        tmp_part_func_sets.part_funcs_4_bpas_on_mls.part_func = sum_4_bpas_on_mls;
        sumormax(
          &mut sum_on_mls,
          tmp_part_func_sets.part_funcs_on_mls.part_func_4_align,
          is_viterbi,
        );
        sumormax(
          &mut sum_on_mls,
          tmp_part_func_sets.part_funcs_on_mls.part_func_4_insert,
          is_viterbi,
        );
        sumormax(
          &mut sum_on_mls,
          tmp_part_func_sets.part_funcs_on_mls.part_func_4_insert_2,
          is_viterbi,
        );
        tmp_part_func_sets.part_funcs_on_mls.part_func = sum_on_mls;
        tmp_part_func_set_mat.insert(pos_pair, tmp_part_func_sets);
      }
      if produces_access_probs || trains_score_params {
        sumormax(
          &mut sum_4_2loop,
          tmp_part_funcs_4_2loop.part_func_4_align,
          is_viterbi,
        );
        sumormax(
          &mut sum_4_2loop,
          tmp_part_funcs_4_2loop.part_func_4_insert,
          is_viterbi,
        );
        sumormax(
          &mut sum_4_2loop,
          tmp_part_funcs_4_2loop.part_func_4_insert_2,
          is_viterbi,
        );
        tmp_part_funcs_4_2loop.part_func = sum_4_2loop;
        tmp_part_func_set_mat_4_2loop.insert(pos_pair, tmp_part_funcs_4_2loop);
      }
    }
  }
  if (produces_access_probs || trains_score_params) && !is_forward {
    for &u in iter.iter() {
      for &v in iter_2.iter() {
        let pos_pair = (u, v);
        if !is_min_gap_ok(&pos_pair, &pseudo_pos_quadruple, max_gap_num_4_il) {
          continue;
        }
        let skips = cache_is_used && tmp_part_func_set_mat_4_decode.contains_key(&pos_pair);
        if skips && !(produces_access_probs || trains_score_params) {
          continue;
        }
        let mut tmp_part_func_sets_4_decode = TmpPartFuncSets::new();
        let mut tmp_part_funcs_4_2loop_decode = TmpPartFuncs::new();
        let pos_pair_4_align = (u + T::one(), v + T::one());
        match tmp_part_func_set_mat.get(&pos_pair_4_align) {
          Some(tmp_part_func_sets_2) => {
            tmp_part_func_sets_4_decode.part_funcs_on_sa.part_func_4_align = tmp_part_func_sets_2.part_funcs_on_sa.part_func;
            tmp_part_func_sets_4_decode.part_funcs_on_sa_4_ml.part_func_4_align = tmp_part_func_sets_2.part_funcs_on_sa_4_ml.part_func;
            tmp_part_func_sets_4_decode.part_funcs_4_ml.part_func_4_align = tmp_part_func_sets_2.part_funcs_4_ml.part_func;
            tmp_part_func_sets_4_decode.part_funcs_4_first_bpas_on_mls.part_func_4_align = tmp_part_func_sets_2.part_funcs_4_first_bpas_on_mls.part_func;
            tmp_part_func_sets_4_decode.part_funcs_4_bpas_on_mls.part_func_4_align = tmp_part_func_sets_2.part_funcs_4_bpas_on_mls.part_func;
            tmp_part_func_sets_4_decode.part_funcs_on_mls.part_func_4_align = tmp_part_func_sets_2.part_funcs_on_mls.part_func;
            match tmp_part_func_set_mat_4_2loop.get(&pos_pair_4_align) {
              Some(part_funcs) => {
                tmp_part_funcs_4_2loop_decode.part_func_4_align = part_funcs.part_func;
              }, None => {},
            }
            tmp_part_func_sets_4_decode.part_funcs_on_sa.part_func_4_insert = tmp_part_func_sets_2.part_funcs_on_sa.part_func_4_align;
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_on_sa.part_func_4_insert,
              tmp_part_func_sets_2.part_funcs_on_sa.part_func_4_insert_2,
              is_viterbi,
            );
            tmp_part_func_sets_4_decode.part_funcs_on_sa_4_ml.part_func_4_insert = tmp_part_func_sets_2.part_funcs_on_sa_4_ml.part_func_4_align;
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_on_sa_4_ml.part_func_4_insert,
              tmp_part_func_sets_2.part_funcs_on_sa_4_ml.part_func_4_insert_2,
              is_viterbi,
            );
            tmp_part_func_sets_4_decode.part_funcs_4_ml.part_func_4_insert = tmp_part_func_sets_2.part_funcs_4_ml.part_func_4_align;
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_4_ml.part_func_4_insert,
              tmp_part_func_sets_2.part_funcs_4_ml.part_func_4_insert_2,
              is_viterbi,
            );
            tmp_part_func_sets_4_decode.part_funcs_4_first_bpas_on_mls.part_func_4_insert = tmp_part_func_sets_2.part_funcs_4_first_bpas_on_mls.part_func_4_align;
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_4_first_bpas_on_mls.part_func_4_insert,
              tmp_part_func_sets_2.part_funcs_4_first_bpas_on_mls.part_func_4_insert_2,
              is_viterbi,
            );
            tmp_part_func_sets_4_decode.part_funcs_4_bpas_on_mls.part_func_4_insert = tmp_part_func_sets_2.part_funcs_4_bpas_on_mls.part_func_4_align;
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_4_bpas_on_mls.part_func_4_insert,
              tmp_part_func_sets_2.part_funcs_4_bpas_on_mls.part_func_4_insert_2,
              is_viterbi,
            );
            tmp_part_func_sets_4_decode.part_funcs_on_mls.part_func_4_insert = tmp_part_func_sets_2.part_funcs_on_mls.part_func_4_align;
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_on_mls.part_func_4_insert,
              tmp_part_func_sets_2.part_funcs_on_mls.part_func_4_insert_2,
              is_viterbi,
            );
            match tmp_part_func_set_mat_4_2loop.get(&pos_pair_4_align) {
              Some(part_funcs) => {
                tmp_part_funcs_4_2loop_decode.part_func_4_insert = part_funcs.part_func_4_align;
                sumormax(
                  &mut tmp_part_funcs_4_2loop_decode.part_func_4_insert,
                  part_funcs.part_func_4_insert_2,
                  is_viterbi,
                );
                tmp_part_funcs_4_2loop_decode.part_func_4_insert_2 = part_funcs.part_func_4_align;
                sumormax(
                  &mut tmp_part_funcs_4_2loop_decode.part_func_4_insert_2,
                  part_funcs.part_func_4_insert,
                  is_viterbi,
                );
              }, None => {},
            }
            tmp_part_func_sets_4_decode.part_funcs_on_sa.part_func_4_insert_2 = tmp_part_func_sets_2.part_funcs_on_sa.part_func_4_align;
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_on_sa.part_func_4_insert_2,
              tmp_part_func_sets_2.part_funcs_on_sa.part_func_4_insert,
              is_viterbi,
            );
            tmp_part_func_sets_4_decode.part_funcs_on_sa_4_ml.part_func_4_insert_2 = tmp_part_func_sets_2.part_funcs_on_sa_4_ml.part_func_4_align;
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_on_sa_4_ml.part_func_4_insert_2,
              tmp_part_func_sets_2.part_funcs_on_sa_4_ml.part_func_4_insert,
              is_viterbi,
            );
            tmp_part_func_sets_4_decode.part_funcs_4_ml.part_func_4_insert_2 = tmp_part_func_sets_2.part_funcs_4_ml.part_func_4_align;
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_4_ml.part_func_4_insert_2,
              tmp_part_func_sets_2.part_funcs_4_ml.part_func_4_insert,
              is_viterbi,
            );
            tmp_part_func_sets_4_decode.part_funcs_4_first_bpas_on_mls.part_func_4_insert_2 = tmp_part_func_sets_2.part_funcs_4_first_bpas_on_mls.part_func_4_align;
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_4_first_bpas_on_mls.part_func_4_insert_2,
              tmp_part_func_sets_2.part_funcs_4_first_bpas_on_mls.part_func_4_insert,
              is_viterbi,
            );
            tmp_part_func_sets_4_decode.part_funcs_4_bpas_on_mls.part_func_4_insert_2 = tmp_part_func_sets_2.part_funcs_4_bpas_on_mls.part_func_4_align;
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_4_bpas_on_mls.part_func_4_insert_2,
              tmp_part_func_sets_2.part_funcs_4_bpas_on_mls.part_func_4_insert,
              is_viterbi,
            );
            tmp_part_func_sets_4_decode.part_funcs_on_mls.part_func_4_insert_2 = tmp_part_func_sets_2.part_funcs_on_mls.part_func_4_align;
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_on_mls.part_func_4_insert_2,
              tmp_part_func_sets_2.part_funcs_on_mls.part_func_4_insert,
              is_viterbi,
            );
          }, None => {},
        }
        let pos_pair_4_insert = (u + T::one(), v);
        match tmp_part_func_set_mat_4_decode.get(&pos_pair_4_insert) {
          Some(tmp_part_func_sets_2) => {
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_on_sa.part_func_4_insert,
              tmp_part_func_sets_2.part_funcs_on_sa.part_func_4_insert + feature_score_sets.extending_gap_count,
              is_viterbi,
            );
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_on_sa_4_ml.part_func_4_insert,
              tmp_part_func_sets_2.part_funcs_on_sa_4_ml.part_func_4_insert + feature_score_sets.extending_gap_count + feature_score_sets.multi_loop_accessible_baseunpairing_count,
              is_viterbi,
            );
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_4_ml.part_func_4_insert,
              tmp_part_func_sets_2.part_funcs_4_ml.part_func_4_insert + feature_score_sets.extending_gap_count + feature_score_sets.multi_loop_accessible_baseunpairing_count,
              is_viterbi,
            );
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_4_first_bpas_on_mls.part_func_4_insert,
              tmp_part_func_sets_2.part_funcs_4_first_bpas_on_mls.part_func_4_insert + feature_score_sets.extending_gap_count + feature_score_sets.multi_loop_accessible_baseunpairing_count,
              is_viterbi,
            );
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_4_bpas_on_mls.part_func_4_insert,
              tmp_part_func_sets_2.part_funcs_4_bpas_on_mls.part_func_4_insert + feature_score_sets.extending_gap_count + feature_score_sets.multi_loop_accessible_baseunpairing_count,
              is_viterbi,
            );
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_on_mls.part_func_4_insert,
              tmp_part_func_sets_2.part_funcs_on_mls.part_func_4_insert + feature_score_sets.extending_gap_count + feature_score_sets.multi_loop_accessible_baseunpairing_count,
              is_viterbi,
            );
            match tmp_part_func_set_mat_4_2loop_decode.get(&pos_pair_4_insert) {
              Some(part_funcs_4_2loop_2) => {
                sumormax(
                  &mut tmp_part_funcs_4_2loop_decode.part_func_4_insert,
                  part_funcs_4_2loop_2.part_func_4_insert + feature_score_sets.extending_gap_count,
                  is_viterbi,
                );
              }, None => {},
            }
          }, None => {},
        }
        let pos_pair_4_insert_2 = (u, v + T::one());
        match tmp_part_func_set_mat_4_decode.get(&pos_pair_4_insert_2) {
          Some(tmp_part_func_sets_2) => {
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_on_sa.part_func_4_insert_2,
              tmp_part_func_sets_2.part_funcs_on_sa.part_func_4_insert_2 + feature_score_sets.extending_gap_count,
              is_viterbi,
            );
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_on_sa_4_ml.part_func_4_insert_2,
              tmp_part_func_sets_2.part_funcs_on_sa_4_ml.part_func_4_insert_2 + feature_score_sets.extending_gap_count + feature_score_sets.multi_loop_accessible_baseunpairing_count,
              is_viterbi,
            );
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_4_ml.part_func_4_insert_2,
              tmp_part_func_sets_2.part_funcs_4_ml.part_func_4_insert_2 + feature_score_sets.extending_gap_count + feature_score_sets.multi_loop_accessible_baseunpairing_count,
              is_viterbi,
            );
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_4_first_bpas_on_mls.part_func_4_insert_2,
              tmp_part_func_sets_2.part_funcs_4_first_bpas_on_mls.part_func_4_insert_2 + feature_score_sets.extending_gap_count + feature_score_sets.multi_loop_accessible_baseunpairing_count,
              is_viterbi,
            );
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_4_bpas_on_mls.part_func_4_insert_2,
              tmp_part_func_sets_2.part_funcs_4_bpas_on_mls.part_func_4_insert_2 + feature_score_sets.extending_gap_count + feature_score_sets.multi_loop_accessible_baseunpairing_count,
              is_viterbi,
            );
            sumormax(
              &mut tmp_part_func_sets_4_decode.part_funcs_on_mls.part_func_4_insert_2,
              tmp_part_func_sets_2.part_funcs_on_mls.part_func_4_insert_2 + feature_score_sets.extending_gap_count + feature_score_sets.multi_loop_accessible_baseunpairing_count,
              is_viterbi,
            );
            match tmp_part_func_set_mat_4_2loop_decode.get(&pos_pair_4_insert_2) {
              Some(part_funcs_4_2loop_2) => {
                sumormax(
                  &mut tmp_part_funcs_4_2loop_decode.part_func_4_insert_2,
                  part_funcs_4_2loop_2.part_func_4_insert_2 + feature_score_sets.extending_gap_count,
                  is_viterbi,
                );
              }, None => {},
            }
          }, None => {},
        }
        tmp_part_func_set_mat_4_decode.insert(pos_pair, tmp_part_func_sets_4_decode);
        tmp_part_func_set_mat_4_2loop_decode.insert(pos_pair, tmp_part_funcs_4_2loop_decode);
      }
    }
  }
  let ref tmp_part_funcs = tmp_part_func_set_mat[&if is_forward {
    rightmost_pos_pair
  } else {
    leftmost_pos_pair
  }];
  let edge_part_func_on_sa = tmp_part_funcs.part_funcs_on_sa.part_func;
  let edge_part_func_4_ml = tmp_part_funcs.part_funcs_4_ml.part_func;
  (
    tmp_part_func_set_mat,
    edge_part_func_on_sa,
    edge_part_func_4_ml,
    tmp_part_func_set_mat_4_2loop,
    tmp_part_func_set_mat_4_decode,
    tmp_part_func_set_mat_4_2loop_decode,
  )
}

pub fn get_sta_prob_mats<T>(
  seq_pair: &SeqPair,
  feature_score_sets: &FeatureCountSets,
  max_bp_span_pair: &PosPair<T>,
  max_gap_num: T,
  max_gap_num_4_il: T,
  sta_part_func_mats: &StaPartFuncMats<T>,
  bpp_mat_pair: &ProbMatPair<T>,
  produces_access_probs: bool,
  global_part_func: PartFunc,
  trains_score_params: bool,
  expected_feature_count_sets: &mut FeatureCountSets,
) -> StaProbMats<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord,
{
  let is_viterbi = false;
  let seq_len_pair = (
    T::from_usize(seq_pair.0.len()).unwrap(),
    T::from_usize(seq_pair.1.len()).unwrap(),
  );
  let pseudo_pos_quadruple = (
    T::zero(),
    seq_len_pair.0 - T::one(),
    T::zero(),
    seq_len_pair.1 - T::one(),
  );
  let mut sta_outside_part_func_4d_mat_4_bpas = PartFunc4dMat::<T>::default();
  let mut sta_prob_mats = StaProbMats::<T>::new(&seq_len_pair);
  for substr_len_1 in range_inclusive(
    T::from_usize(CONSPROB_MIN_HAIRPIN_LOOP_SPAN).unwrap(),
    max_bp_span_pair.0,
  )
  .rev()
  {
    for i in range(T::one(), seq_len_pair.0 - substr_len_1) {
      let j = i + substr_len_1 - T::one();
      let (long_i, long_j) = (i.to_usize().unwrap(), j.to_usize().unwrap());
      let base_pair = (seq_pair.0[long_i], seq_pair.0[long_j]);
      let pos_pair = (i, j);
      if !is_canonical(&base_pair) {
        continue;
      }
      if !bpp_mat_pair.0.contains_key(&pos_pair) {
        continue;
      }
      let dict_min_base_pair = get_dict_min_base_pair(&base_pair);
      let mismatch_pair = (seq_pair.0[long_j + 1], seq_pair.0[long_i - 1]);
      for substr_len_2 in range_inclusive(
        T::from_usize(CONSPROB_MIN_HAIRPIN_LOOP_LEN).unwrap(),
        max_bp_span_pair.1,
      )
      .rev()
      {
        for k in range(T::one(), seq_len_pair.1 - substr_len_2) {
          let l = k + substr_len_2 - T::one();
          if !is_min_gap_ok(&(i, k), &pseudo_pos_quadruple, max_gap_num_4_il) {
            continue;
          }
          if !is_min_gap_ok(&(j, l), &pseudo_pos_quadruple, max_gap_num_4_il) {
            continue;
          }
          let (long_k, long_l) = (k.to_usize().unwrap(), l.to_usize().unwrap());
          let pos_quadruple = (i, j, k, l);
          let base_pair_2 = (seq_pair.1[long_k], seq_pair.1[long_l]);
          if !is_canonical(&base_pair_2) {
            continue;
          }
          let dict_min_base_pair_2 = get_dict_min_base_pair(&base_pair_2);
          let mismatch_pair_2 = (seq_pair.1[long_l + 1], seq_pair.1[long_k - 1]);
          let pos_pair_2 = (k, l);
          match sta_part_func_mats
            .part_func_4d_mat_4_bpas
            .get(&pos_quadruple)
          {
            Some(&part_func_4_bpa) => {
              let prob_coeff = part_func_4_bpa - global_part_func;
              let mut sum = NEG_INFINITY;
              let mut forward_term = sum;
              let mut backward_term = sum;
              match sta_part_func_mats
                .forward_part_func_set_mat_4_external_loop
                .get(&(i - T::one(), k - T::one()))
              {
                Some(part_funcs) => {
                  forward_term = part_funcs.part_func;
                }
                None => {}
              }
              match sta_part_func_mats
                .backward_part_func_set_mat_4_external_loop
                .get(&(j + T::one(), l + T::one()))
              {
                Some(part_funcs) => {
                  backward_term =
                    part_funcs.part_func;
                }
                None => {}
              }
              let part_func_4_el = forward_term + backward_term;
              if part_func_4_el > NEG_INFINITY {
                let coefficient = sta_part_func_mats.part_func_4d_mat_4_bpas_accessible_on_els
                  [&pos_quadruple]
                  - part_func_4_bpa;
                sum = coefficient + part_func_4_el;
                let bpap_4_el = prob_coeff + sum;
                if produces_access_probs {
                  match sta_prob_mats.bpp_mat_pair_4_el.0.get_mut(&pos_pair) {
                    Some(bpp_4_el) => {
                      sumormax(bpp_4_el, bpap_4_el, is_viterbi);
                    }
                    None => {
                      sta_prob_mats
                        .bpp_mat_pair_4_el
                        .0
                        .insert(pos_pair, bpap_4_el);
                    }
                  }
                  match sta_prob_mats.bpp_mat_pair_4_el.1.get_mut(&pos_pair_2) {
                    Some(bpp_4_el) => {
                      sumormax(bpp_4_el, bpap_4_el, is_viterbi);
                    }
                    None => {
                      sta_prob_mats
                        .bpp_mat_pair_4_el
                        .1
                        .insert(pos_pair_2, bpap_4_el);
                    }
                  }
                }
                if trains_score_params {
                  // Count external loop accessible basepairings.
                  sumormax(&mut expected_feature_count_sets
                    .external_loop_accessible_basepairing_count, (2. as Prob).ln() + bpap_4_el, is_viterbi);
                  // Count helix ends.
                  sumormax(&mut expected_feature_count_sets.helix_end_count_mat[base_pair.1][base_pair.0], bpap_4_el, is_viterbi);
                  sumormax(&mut expected_feature_count_sets.helix_end_count_mat[base_pair_2.1][base_pair_2.0], bpap_4_el, is_viterbi);
                  // Count external loop terminal mismatches.
                  if j < seq_len_pair.0 - T::from_usize(2).unwrap() {
                    sumormax(&mut expected_feature_count_sets.left_dangle_count_mat
                      [base_pair.1][base_pair.0][seq_pair.0[long_j + 1]], bpap_4_el, is_viterbi);
                  }
                  if i > T::one() {
                    sumormax(&mut expected_feature_count_sets.right_dangle_count_mat
                      [base_pair.1][base_pair.0][seq_pair.0[long_i - 1]], bpap_4_el, is_viterbi);
                  }
                  if l < seq_len_pair.1 - T::from_usize(2).unwrap() {
                    sumormax(&mut expected_feature_count_sets.left_dangle_count_mat
                      [base_pair_2.1][base_pair_2.0][seq_pair.1[long_l + 1]], bpap_4_el, is_viterbi);
                  }
                  if k > T::one() {
                    sumormax(&mut expected_feature_count_sets.right_dangle_count_mat
                      [base_pair_2.1][base_pair_2.0][seq_pair.1[long_k - 1]], bpap_4_el, is_viterbi);
                  }
                }
              }
              for m in range(T::one(), i) {
                let long_m = m.to_usize().unwrap();
                for n in range(j + T::one(), seq_len_pair.0 - T::one()) {
                  let long_n = n.to_usize().unwrap();
                  let base_pair_3 = (seq_pair.0[long_m], seq_pair.0[long_n]);
                  if !is_canonical(&base_pair_3) {
                    continue;
                  }
                  let pos_pair_3 = (m, n);
                  let loop_len_pair = (long_i - long_m - 1, long_n - long_j - 1);
                  if loop_len_pair.0 + loop_len_pair.1 > CONSPROB_MAX_TWOLOOP_LEN {
                    continue;
                  }
                  if !bpp_mat_pair.0.contains_key(&pos_pair_3) {
                    continue;
                  }
                  let is_stack = loop_len_pair.0 == 0 && loop_len_pair.1 == 0;
                  let is_bulge_loop = (loop_len_pair.0 == 0 || loop_len_pair.1 == 0) && !is_stack;
                  let mismatch_pair_3 = (seq_pair.0[long_m + 1], seq_pair.0[long_n - 1]);
                  let twoloop_score = get_consprob_twoloop_score(
                    feature_score_sets,
                    seq_pair.0,
                    &(long_m, long_n),
                    &(long_i, long_j),
                  );
                  for o in range(T::one(), k) {
                    if !is_min_gap_ok(&(m, o), &pseudo_pos_quadruple, max_gap_num_4_il) {
                      continue;
                    }
                    let long_o = o.to_usize().unwrap();
                    for p in range(l + T::one(), seq_len_pair.1 - T::one()) {
                      if !is_min_gap_ok(&(n, p), &pseudo_pos_quadruple, max_gap_num_4_il) {
                        continue;
                      }
                      let long_p = p.to_usize().unwrap();
                      let base_pair_4 = (seq_pair.1[long_o], seq_pair.1[long_p]);
                      if !is_canonical(&base_pair_4) {
                        continue;
                      }
                      let loop_len_pair_2 = (long_k - long_o - 1, long_p - long_l - 1);
                      if loop_len_pair_2.0 + loop_len_pair_2.1 > CONSPROB_MAX_TWOLOOP_LEN {
                        continue;
                      }
                      let pos_quadruple_2 = (m, n, o, p);
                      match sta_outside_part_func_4d_mat_4_bpas.get(&pos_quadruple_2) {
                        Some(&part_func) => {
                          let is_stack_2 = loop_len_pair_2.0 == 0 && loop_len_pair_2.1 == 0;
                          let is_bulge_loop_2 =
                            (loop_len_pair_2.0 == 0 || loop_len_pair_2.1 == 0) && !is_stack_2;
                          let mismatch_pair_4 = (seq_pair.1[long_o + 1], seq_pair.1[long_p - 1]);
                          let ref forward_tmp_part_func_set_mat = sta_part_func_mats
                            .forward_tmp_part_func_set_mats_with_pos_pairs[&(m, o)];
                          let ref backward_tmp_part_func_set_mat = sta_part_func_mats
                            .backward_tmp_part_func_set_mats_with_pos_pairs[&(n, p)];
                          let mut forward_term = NEG_INFINITY;
                          let mut backward_term = forward_term;
                          match forward_tmp_part_func_set_mat.get(&(i - T::one(), k - T::one())) {
                            Some(part_func_sets) => {
                              let ref part_funcs = part_func_sets.part_funcs_on_sa;
                              forward_term = part_funcs.part_func;
                            }
                            None => {}
                          }
                          match backward_tmp_part_func_set_mat.get(&(j + T::one(), l + T::one())) {
                            Some(part_func_sets) => {
                              let ref part_funcs = part_func_sets.part_funcs_on_sa;
                              backward_term = part_funcs.part_func;
                            }
                            None => {}
                          }
                          let part_func_4_2l = forward_term + backward_term;
                          if part_func_4_2l > NEG_INFINITY {
                            let basepair_align_score = feature_score_sets.basepair_align_count_mat[base_pair_3.0][base_pair_3.1][base_pair_4.0][base_pair_4.1];
                            let twoloop_score_2 = get_consprob_twoloop_score(
                              feature_score_sets,
                              seq_pair.1,
                              &(long_o, long_p),
                              &(long_k, long_l),
                            );
                            let coefficient = basepair_align_score + twoloop_score + twoloop_score_2 + part_func;
                            let part_func_4_2l = coefficient + part_func_4_2l;
                            sumormax(&mut sum, part_func_4_2l, is_viterbi);
                            let bpap_4_2l = prob_coeff + part_func_4_2l;
                            if produces_access_probs {
                              match sta_prob_mats.access_bpp_mat_pair_4_2l.0.get_mut(&pos_pair) {
                                Some(bpp_4_2l) => {
                                  sumormax(bpp_4_2l, bpap_4_2l, is_viterbi);
                                }
                                None => {
                                  sta_prob_mats
                                    .access_bpp_mat_pair_4_2l
                                    .0
                                    .insert(pos_pair, bpap_4_2l);
                                }
                              }
                              match sta_prob_mats
                                .access_bpp_mat_pair_4_2l
                                .1
                                .get_mut(&pos_pair_2)
                              {
                                Some(bpp_4_2l) => {
                                  sumormax(bpp_4_2l, bpap_4_2l, is_viterbi);
                                }
                                None => {
                                  sta_prob_mats
                                    .access_bpp_mat_pair_4_2l
                                    .1
                                    .insert(pos_pair_2, bpap_4_2l);
                                }
                              }
                            }
                            if trains_score_params {
                              if is_stack {
                                // Count a stack.
                                let dict_min_stack = get_dict_min_stack(&base_pair_3, &base_pair);
                                sumormax(&mut expected_feature_count_sets.stack_count_mat[dict_min_stack.0.0]
                                  [dict_min_stack.0.1][dict_min_stack.1.0][dict_min_stack.1.1], bpap_4_2l, is_viterbi);
                              } else {
                                if is_bulge_loop {
                                  // Count a bulge loop length.
                                  let bulge_loop_len = if loop_len_pair.0 == 0 {
                                    loop_len_pair.1
                                  } else {
                                    loop_len_pair.0
                                  };
                                  sumormax(&mut expected_feature_count_sets.bulge_loop_length_counts
                                    [bulge_loop_len - 1], bpap_4_2l, is_viterbi);
                                  // Count a 0x1 bulge loop.
                                  if bulge_loop_len == 1 {
                                    let mismatch = if loop_len_pair.0 == 0 {mismatch_pair_3.1} else {mismatch_pair_3.0};
                                    sumormax(&mut expected_feature_count_sets.bulge_loop_0x1_length_counts
                                      [mismatch], bpap_4_2l, is_viterbi);
                                  }
                                } else {
                                  // Count an interior loop length.
                                  sumormax(&mut expected_feature_count_sets.interior_loop_length_counts
                                    [loop_len_pair.0 + loop_len_pair.1 - 2], bpap_4_2l, is_viterbi);
                                  let diff = get_diff(loop_len_pair.0, loop_len_pair.1);
                                  if diff == 0 {
                                    sumormax(&mut expected_feature_count_sets.interior_loop_length_counts_symm
                                      [loop_len_pair.0 - 1], bpap_4_2l, is_viterbi);
                                  } else {
                                    sumormax(&mut expected_feature_count_sets.interior_loop_length_counts_asymm
                                      [diff - 1], bpap_4_2l, is_viterbi);
                                  }
                                  // Count a 1x1 interior loop.
                                  if loop_len_pair.0 == 1 && loop_len_pair.1 == 1 {
                                    let dict_min_mismatch_pair_3 = get_dict_min_mismatch_pair(&mismatch_pair_3);
                                    sumormax(&mut expected_feature_count_sets.interior_loop_1x1_length_count_mat
                                      [dict_min_mismatch_pair_3.0][dict_min_mismatch_pair_3.1], bpap_4_2l, is_viterbi);
                                  }
                                  // Count an explicit interior loop length pair.
                                  if loop_len_pair.0 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT && loop_len_pair.1 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT {
                                    let dict_min_loop_len_pair =  get_dict_min_loop_len_pair(&loop_len_pair);
                                    sumormax(&mut expected_feature_count_sets.interior_loop_length_count_mat_explicit
                                      [dict_min_loop_len_pair.0 - 1][dict_min_loop_len_pair.1 - 1], bpap_4_2l, is_viterbi);
                                  }
                                }
                                // Count helix ends.
                                sumormax(&mut expected_feature_count_sets.helix_end_count_mat[base_pair.1][base_pair.0], bpap_4_2l, is_viterbi);
                                sumormax(&mut expected_feature_count_sets.helix_end_count_mat[base_pair_3.0][base_pair_3.1], bpap_4_2l, is_viterbi);
                                // Count 2-loop terminal mismatches.
                                sumormax(&mut expected_feature_count_sets.terminal_mismatch_count_mat
                                  [base_pair_3.0][base_pair_3.1][mismatch_pair_3.0]
                                  [mismatch_pair_3.1], bpap_4_2l, is_viterbi);
                                sumormax(&mut expected_feature_count_sets.terminal_mismatch_count_mat
                                  [base_pair.1][base_pair.0][mismatch_pair.0][mismatch_pair.1],
                                  bpap_4_2l, is_viterbi);
                              }
                              if is_stack_2 {
                                // Count a stack.
                                let dict_min_stack_2 = get_dict_min_stack(&base_pair_4, &base_pair_2);
                                sumormax(&mut expected_feature_count_sets.stack_count_mat[dict_min_stack_2.0.0]
                                  [dict_min_stack_2.0.1][dict_min_stack_2.1.0][dict_min_stack_2.1.1], bpap_4_2l, is_viterbi);
                              } else {
                                if is_bulge_loop_2 {
                                  // Count a bulge loop length.
                                  let bulge_loop_len_2 = if loop_len_pair_2.0 == 0 {
                                    loop_len_pair_2.1
                                  } else {
                                    loop_len_pair_2.0
                                  };
                                  sumormax(&mut expected_feature_count_sets.bulge_loop_length_counts
                                    [bulge_loop_len_2 - 1], bpap_4_2l, is_viterbi);
                                  // Count a 0x1 bulge loop.
                                  if bulge_loop_len_2 == 1 {
                                    let mismatch_2 = if loop_len_pair_2.0 == 0 {mismatch_pair_4.1} else {mismatch_pair_4.0};
                                    sumormax(&mut expected_feature_count_sets.bulge_loop_0x1_length_counts
                                      [mismatch_2], bpap_4_2l, is_viterbi);
                                  }
                                } else {
                                  // Count an interior loop length.
                                  sumormax(&mut expected_feature_count_sets.interior_loop_length_counts
                                    [loop_len_pair_2.0 + loop_len_pair_2.1 - 2], bpap_4_2l, is_viterbi);
                                  let diff_2 = get_diff(loop_len_pair_2.0, loop_len_pair_2.1);
                                  if diff_2 == 0 {
                                    sumormax(&mut expected_feature_count_sets.interior_loop_length_counts_symm
                                      [loop_len_pair_2.0 - 1], bpap_4_2l, is_viterbi);
                                  } else {
                                    sumormax(&mut expected_feature_count_sets.interior_loop_length_counts_asymm
                                      [diff_2 - 1], bpap_4_2l, is_viterbi);
                                  }
                                  // Count a 1x1 interior loop.
                                  if loop_len_pair_2.0 == 1 && loop_len_pair_2.1 == 1 {
                                    let dict_min_mismatch_pair_4 = get_dict_min_mismatch_pair(&mismatch_pair_4);
                                    sumormax(&mut expected_feature_count_sets.interior_loop_1x1_length_count_mat
                                      [dict_min_mismatch_pair_4.0][dict_min_mismatch_pair_4.1], bpap_4_2l, is_viterbi);
                                  }
                                  // Count an explicit interior loop length pair.
                                  if loop_len_pair_2.0 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT && loop_len_pair_2.1 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT {
                                    let dict_min_loop_len_pair_2 =  get_dict_min_loop_len_pair(&loop_len_pair_2);
                                    sumormax(&mut expected_feature_count_sets.interior_loop_length_count_mat_explicit
                                      [dict_min_loop_len_pair_2.0 - 1][dict_min_loop_len_pair_2.1 - 1], bpap_4_2l, is_viterbi);
                                  }
                                }
                                // Count helix ends.
                                sumormax(&mut expected_feature_count_sets.helix_end_count_mat[base_pair_2.1][base_pair_2.0], bpap_4_2l, is_viterbi);
                                sumormax(&mut expected_feature_count_sets.helix_end_count_mat[base_pair_4.0][base_pair_4.1], bpap_4_2l, is_viterbi);
                                // Count 2-loop terminal mismatches.
                                sumormax(&mut expected_feature_count_sets.terminal_mismatch_count_mat
                                  [base_pair_4.0][base_pair_4.1][mismatch_pair_4.0]
                                  [mismatch_pair_4.1], bpap_4_2l, is_viterbi);
                                sumormax(&mut expected_feature_count_sets.terminal_mismatch_count_mat
                                  [base_pair_2.1][base_pair_2.0][mismatch_pair_2.0]
                                  [mismatch_pair_2.1], bpap_4_2l, is_viterbi);
                              }
                            }
                          }
                        }
                        None => {}
                      }
                    }
                  }
                }
              }
              let part_func_ratio = sta_part_func_mats.part_func_4d_mat_4_bpas_accessible_on_mls
                [&pos_quadruple]
                - part_func_4_bpa;
              for m in range(T::one(), i) {
                let long_m = m.to_usize().unwrap();
                for n in range(j + T::one(), seq_len_pair.0 - T::one()) {
                  let long_n = n.to_usize().unwrap();
                  let base_pair_3 = (seq_pair.0[long_m], seq_pair.0[long_n]);
                  if !is_canonical(&base_pair_3) {
                    continue;
                  }
                  let pos_pair_3 = (m, n);
                  if !bpp_mat_pair.0.contains_key(&pos_pair_3) {
                    continue;
                  }
                  let mismatch_pair_3 = (seq_pair.0[long_m + 1], seq_pair.0[long_n - 1]);
                  let base_pair = (seq_pair.0[long_m], seq_pair.0[long_n]);
                  let multi_loop_closing_basepairing_score = get_consprob_multi_loop_closing_basepairing_score(
                    feature_score_sets,
                    seq_pair.0,
                    &(long_m, long_n),
                  );
                  for o in range(T::one(), k) {
                    if !is_min_gap_ok(&(m, o), &pseudo_pos_quadruple, max_gap_num_4_il) {
                      continue;
                    }
                    let long_o = o.to_usize().unwrap();
                    for p in range(l + T::one(), seq_len_pair.1 - T::one()) {
                      if !is_min_gap_ok(&(n, p), &pseudo_pos_quadruple, max_gap_num_4_il) {
                        continue;
                      }
                      let long_p = p.to_usize().unwrap();
                      let base_pair_4 = (seq_pair.1[long_o], seq_pair.1[long_p]);
                      if !is_canonical(&base_pair_4) {
                        continue;
                      }
                      let pos_quadruple_2 = (m, n, o, p);
                      match sta_outside_part_func_4d_mat_4_bpas.get(&pos_quadruple_2) {
                        Some(&part_func_4_bpa_2) => {
                          let mismatch_pair_4 = (seq_pair.1[long_o + 1], seq_pair.1[long_p - 1]);
                          let base_pair_2 = (seq_pair.1[long_o], seq_pair.1[long_p]);
                          let ref forward_tmp_part_func_set_mat = sta_part_func_mats
                            .forward_tmp_part_func_set_mats_with_pos_pairs[&(m, o)];
                          let ref backward_tmp_part_func_set_mat = sta_part_func_mats
                            .backward_tmp_part_func_set_mats_with_pos_pairs[&(n, p)];
                          let basepair_align_score = feature_score_sets.basepair_align_count_mat[base_pair_3.0][base_pair_3.1][base_pair_4.0][base_pair_4.1];
                          let multi_loop_closing_basepairing_score_2 = get_consprob_multi_loop_closing_basepairing_score(
                            feature_score_sets,
                            seq_pair.1,
                            &(long_o, long_p),
                          );
                          let mut forward_term = NEG_INFINITY;
                          let mut forward_term_2 = forward_term;
                          let mut backward_term = forward_term;
                          let mut backward_term_2 = forward_term;
                          match forward_tmp_part_func_set_mat.get(&(i - T::one(), k - T::one())) {
                            Some(part_func_sets) => {
                              forward_term = part_func_sets.part_funcs_4_bpas_on_mls.part_func;
                              forward_term_2 = part_func_sets.part_funcs_on_sa_4_ml.part_func;
                            }
                            None => {}
                          }
                          match backward_tmp_part_func_set_mat.get(&(j + T::one(), l + T::one())) {
                            Some(part_func_sets) => {
                              backward_term = part_func_sets.part_funcs_on_mls.part_func;
                              backward_term_2 = part_func_sets.part_funcs_4_bpas_on_mls.part_func;
                            }
                            None => {}
                          }
                          let mut part_func_4_ml = forward_term + backward_term;
                          sumormax(
                            &mut part_func_4_ml,
                            forward_term_2 + backward_term_2,
                            is_viterbi,
                          );
                          if part_func_4_ml > NEG_INFINITY {
                            let coefficient = part_func_ratio
                              + basepair_align_score
                              + multi_loop_closing_basepairing_score
                              + multi_loop_closing_basepairing_score_2
                              + part_func_4_bpa_2;
                            let part_func_4_ml = coefficient + part_func_4_ml;
                            sumormax(&mut sum, part_func_4_ml, is_viterbi);
                            let bpap_4_ml = prob_coeff + part_func_4_ml;
                            if produces_access_probs {
                              match sta_prob_mats.access_bpp_mat_pair_4_ml.0.get_mut(&pos_pair) {
                                Some(bpp_4_ml) => {
                                  sumormax(bpp_4_ml, bpap_4_ml, is_viterbi);
                                }
                                None => {
                                  sta_prob_mats
                                    .access_bpp_mat_pair_4_ml
                                    .0
                                    .insert(pos_pair, bpap_4_ml);
                                }
                              }
                              match sta_prob_mats
                                .access_bpp_mat_pair_4_ml
                                .1
                                .get_mut(&pos_pair_2)
                              {
                                Some(bpp_4_ml) => {
                                  sumormax(bpp_4_ml, bpap_4_ml, is_viterbi);
                                }
                                None => {
                                  sta_prob_mats
                                    .access_bpp_mat_pair_4_ml
                                    .1
                                    .insert(pos_pair_2, bpap_4_ml);
                                }
                              }
                            }
                            if trains_score_params {
                              // Count multi-loop terminam mismatches.
                              sumormax(&mut expected_feature_count_sets.left_dangle_count_mat
                                [base_pair_3.0][base_pair_3.1][mismatch_pair_3.0]
                                , bpap_4_ml, is_viterbi);
                              sumormax(&mut expected_feature_count_sets.right_dangle_count_mat
                                [base_pair_3.0][base_pair_3.1][mismatch_pair_3.1]
                                , bpap_4_ml, is_viterbi);
                              sumormax(&mut expected_feature_count_sets.left_dangle_count_mat[base_pair_4.0]
                                [base_pair_4.1][mismatch_pair_4.0], bpap_4_ml, is_viterbi);
                              sumormax(&mut expected_feature_count_sets.left_dangle_count_mat[base_pair_4.0]
                                [base_pair_4.1][mismatch_pair_4.1], bpap_4_ml, is_viterbi);
                              sumormax(&mut expected_feature_count_sets.left_dangle_count_mat
                                [base_pair.1][base_pair.0][mismatch_pair.0]
                                , bpap_4_ml, is_viterbi);
                              sumormax(&mut expected_feature_count_sets.right_dangle_count_mat
                                [base_pair.1][base_pair.0][mismatch_pair.1]
                                , bpap_4_ml, is_viterbi);
                              sumormax(&mut expected_feature_count_sets.left_dangle_count_mat[base_pair_2.1]
                                [base_pair_2.0][mismatch_pair_2.0], bpap_4_ml, is_viterbi);
                              sumormax(&mut expected_feature_count_sets.right_dangle_count_mat[base_pair_2.1]
                                [base_pair_2.0][mismatch_pair_2.1], bpap_4_ml, is_viterbi);
                              // Count helix ends.
                              sumormax(&mut expected_feature_count_sets.helix_end_count_mat
                                [base_pair_3.0][base_pair_3.1], bpap_4_ml, is_viterbi);
                              sumormax(&mut expected_feature_count_sets.helix_end_count_mat[base_pair_4.0]
                                [base_pair_4.1], bpap_4_ml, is_viterbi);
                              sumormax(&mut expected_feature_count_sets.helix_end_count_mat
                                [base_pair.1][base_pair.0]
                                , bpap_4_ml, is_viterbi);
                              sumormax(&mut expected_feature_count_sets.helix_end_count_mat[base_pair_2.1]
                                [base_pair_2.0], bpap_4_ml, is_viterbi);
                              // Count multi-loop closings.
                              sumormax(&mut expected_feature_count_sets.multi_loop_base_count, (2. as Prob).ln() + bpap_4_ml, is_viterbi);
                              // Count multi-loop closing basepairings.
                              sumormax(&mut expected_feature_count_sets.multi_loop_basepairing_count, (2. as Prob).ln() + bpap_4_ml, is_viterbi);
                              // Count multi-loop accessible basepairings.
                              sumormax(&mut expected_feature_count_sets
                                .multi_loop_basepairing_count, (2. as Prob).ln() + bpap_4_ml, is_viterbi);
                            }
                          }
                        }
                        None => {}
                      }
                    }
                  }
                }
              }
              if sum > NEG_INFINITY {
                sta_outside_part_func_4d_mat_4_bpas.insert(pos_quadruple, sum);
                let bpap = prob_coeff + sum;
                if trains_score_params {
                  // Count base pairs.
                  sumormax(&mut expected_feature_count_sets.base_pair_count_mat[dict_min_base_pair.0][dict_min_base_pair.1], bpap, is_viterbi);
                  sumormax(&mut expected_feature_count_sets.base_pair_count_mat[dict_min_base_pair_2.0][dict_min_base_pair_2.1], bpap, is_viterbi);
                  // Count a basepair alignment.
                  let dict_min_basepair_align = get_dict_min_basepair_align(&base_pair, &base_pair_2);
                  sumormax(&mut expected_feature_count_sets.basepair_align_count_mat[dict_min_basepair_align.0.0][dict_min_basepair_align.0.1]
                    [dict_min_basepair_align.1.0][dict_min_basepair_align.1.1], bpap, is_viterbi);
                }
                let bpap = bpap.exp();
                match sta_prob_mats.bpp_mat_pair.0.get_mut(&(i, j)) {
                  Some(bpp) => {
                    *bpp += bpap;
                  }
                  None => {
                    sta_prob_mats.bpp_mat_pair.0.insert((i, j), bpap);
                  }
                }
                match sta_prob_mats.bpp_mat_pair.1.get_mut(&(k, l)) {
                  Some(bpp) => {
                    *bpp += bpap;
                  }
                  None => {
                    sta_prob_mats.bpp_mat_pair.1.insert((k, l), bpap);
                  }
                }
                sta_prob_mats.upp_mat_pair.0[long_i] -= bpap;
                sta_prob_mats.upp_mat_pair.0[long_j] -= bpap;
                sta_prob_mats.upp_mat_pair.1[long_k] -= bpap;
                sta_prob_mats.upp_mat_pair.1[long_l] -= bpap;
                if trains_score_params {
                  let part_func_on_sa = sta_part_func_mats
                .forward_tmp_part_func_set_mats_with_pos_pairs[&(i, k)][&(j - T::one(), l - T::one())].part_funcs_on_sa.part_func;
                  let hairpin_loop_score = get_consprob_hairpin_loop_score(feature_score_sets, seq_pair.0, &(long_i, long_j));
                  let hairpin_loop_score_2 = get_consprob_hairpin_loop_score(feature_score_sets, seq_pair.1, &(long_k, long_l));
                  let bpap_4_hl = sum - global_part_func + part_func_on_sa + hairpin_loop_score + hairpin_loop_score_2;
                  if bpap_4_hl > NEG_INFINITY {
                    sumormax(
                      &mut expected_feature_count_sets.hairpin_loop_length_counts[long_j - long_i - 1 - CONSPROB_MIN_HAIRPIN_LOOP_LEN],
                      bpap_4_hl,
                      is_viterbi,
                    );
                    sumormax(
                      &mut expected_feature_count_sets.hairpin_loop_length_counts[long_l - long_k - 1 - CONSPROB_MIN_HAIRPIN_LOOP_LEN],
                      bpap_4_hl,
                      is_viterbi,
                    );
                    let mismatch_pair = (seq_pair.0[long_i + 1], seq_pair.0[long_j - 1]);
                    sumormax(&mut expected_feature_count_sets.terminal_mismatch_count_mat
                      [base_pair.0][base_pair.1][mismatch_pair.0]
                      [mismatch_pair.1], bpap_4_hl, is_viterbi);
                    let mismatch_pair_2 = (seq_pair.1[long_k + 1], seq_pair.1[long_l - 1]);
                    sumormax(&mut expected_feature_count_sets.terminal_mismatch_count_mat
                      [base_pair_2.0][base_pair_2.1][mismatch_pair_2.0]
                      [mismatch_pair_2.1], bpap_4_hl, is_viterbi);
                  }
                }
              }
            }
            None => {}
          }
        }
      }
    }
  }
  if produces_access_probs || trains_score_params {
    for u in range_inclusive(T::zero(), seq_len_pair.0 - T::from_usize(2).unwrap()) {
      let long_u = u.to_usize().unwrap();
      let base = seq_pair.0[long_u];
      for v in range_inclusive(T::zero(), seq_len_pair.1 - T::from_usize(2).unwrap()) {
        if u == T::zero() && v == T::zero() {
          continue;
        }
        let pos_pair = (u, v);
        if !is_min_gap_ok(&pos_pair, &pseudo_pos_quadruple, max_gap_num) {
          continue;
        }
        let long_v = v.to_usize().unwrap();
        let base_2 = seq_pair.1[long_v];
        let pos_pair_4_loop_align = (u - T::one(), v - T::one());
        let pos_pair_4_insert = (u - T::one(), v);
        let pos_pair_4_insert_2 = (u, v - T::one());
        let mut backward_term_4_align = NEG_INFINITY;
        let mut backward_term_4_insert = backward_term_4_align;
        let mut backward_term_4_insert_2 = backward_term_4_align;
        match sta_part_func_mats
          .backward_part_func_set_mat_4_external_loop_decode
          .get(&pos_pair)
        {
          Some(part_funcs) => {
            backward_term_4_align = part_funcs.part_func_4_align;
            backward_term_4_insert = part_funcs.part_func_4_insert;
            backward_term_4_insert_2 = part_funcs.part_func_4_insert_2;
          }
          None => {}
        }
        let dict_min_loop_align = get_dict_min_loop_align(&(base, base_2));
        if u > T::zero() && v > T::zero() {
          match sta_part_func_mats
            .forward_part_func_set_mat_4_external_loop
            .get(&pos_pair_4_loop_align)
          {
            Some(part_funcs) => {
              let loop_align_score = feature_score_sets.loop_align_count_mat[base][base_2];
              let loop_align_prob_4_el = loop_align_score + part_funcs.part_func + backward_term_4_align - global_part_func + 2. * feature_score_sets.external_loop_accessible_baseunpairing_count;
              if produces_access_probs {
                sumormax(
                  &mut sta_prob_mats.upp_mat_pair_4_el.0[long_u],
                  loop_align_prob_4_el,
                  is_viterbi,
                );
                sumormax(
                  &mut sta_prob_mats.upp_mat_pair_4_el.1[long_v],
                  loop_align_prob_4_el,
                  is_viterbi,
                );
              }
              if trains_score_params {
                sumormax(
                  &mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1],
                  loop_align_prob_4_el,
                  is_viterbi,
                );
                sumormax(
                  &mut expected_feature_count_sets.external_loop_accessible_baseunpairing_count,
                  (2. as Prob).ln() + loop_align_prob_4_el,
                  is_viterbi,
                );
              }
            }, None => {},
          }
        }
        if u > T::zero() {
          match sta_part_func_mats
            .forward_part_func_set_mat_4_external_loop
            .get(&pos_pair_4_insert)
          {
            Some(part_funcs) => {
              let insert_prob = feature_score_sets.opening_gap_count + part_funcs.part_func_4_align + backward_term_4_insert - global_part_func + feature_score_sets.external_loop_accessible_baseunpairing_count;
              if produces_access_probs {
                sumormax(
                  &mut sta_prob_mats.upp_mat_pair_4_el.0[long_u],
                  insert_prob,
                  is_viterbi,
                );
              }
              if trains_score_params {
                sumormax(
                  &mut expected_feature_count_sets.opening_gap_count,
                  insert_prob,
                  is_viterbi,
                );
                sumormax(
                  &mut expected_feature_count_sets.external_loop_accessible_baseunpairing_count,
                  insert_prob,
                  is_viterbi,
                );
              }
              let insert_prob = feature_score_sets.extending_gap_count + part_funcs.part_func_4_insert + backward_term_4_insert - global_part_func + feature_score_sets.external_loop_accessible_baseunpairing_count;
              if produces_access_probs {
                sumormax(
                  &mut sta_prob_mats.upp_mat_pair_4_el.0[long_u],
                  insert_prob,
                  is_viterbi,
                );
              }
              if trains_score_params {
                sumormax(
                  &mut expected_feature_count_sets.extending_gap_count,
                  insert_prob,
                  is_viterbi,
                );
                sumormax(
                  &mut expected_feature_count_sets.external_loop_accessible_baseunpairing_count,
                  insert_prob,
                  is_viterbi,
                );
              }
              let insert_prob = feature_score_sets.opening_gap_count + part_funcs.part_func_4_insert_2 + backward_term_4_insert - global_part_func + feature_score_sets.external_loop_accessible_baseunpairing_count;
              if produces_access_probs {
                sumormax(
                  &mut sta_prob_mats.upp_mat_pair_4_el.0[long_u],
                  insert_prob,
                  is_viterbi,
                );
              }
              if trains_score_params {
                sumormax(
                  &mut expected_feature_count_sets.opening_gap_count,
                  insert_prob,
                  is_viterbi,
                );
                sumormax(
                  &mut expected_feature_count_sets.external_loop_accessible_baseunpairing_count,
                  insert_prob,
                  is_viterbi,
                );
              }
            },
            None => {},
          }
        }
        if v > T::zero() {
          match sta_part_func_mats
            .forward_part_func_set_mat_4_external_loop
            .get(&pos_pair_4_insert_2)
          {
            Some(part_funcs) => {
              let insert_prob = feature_score_sets.opening_gap_count + part_funcs.part_func_4_align + backward_term_4_insert_2 - global_part_func + feature_score_sets.external_loop_accessible_baseunpairing_count;
              if produces_access_probs {
                sumormax(
                  &mut sta_prob_mats.upp_mat_pair_4_el.1[long_v],
                  insert_prob,
                  is_viterbi,
                );
              }
              if trains_score_params {
                sumormax(
                  &mut expected_feature_count_sets.opening_gap_count,
                  insert_prob,
                  is_viterbi,
                );
                sumormax(
                  &mut expected_feature_count_sets.external_loop_accessible_baseunpairing_count,
                  insert_prob,
                  is_viterbi,
                );
              }
              let insert_prob = feature_score_sets.opening_gap_count + part_funcs.part_func_4_insert + backward_term_4_insert_2 - global_part_func + feature_score_sets.external_loop_accessible_baseunpairing_count;
              if produces_access_probs {
                sumormax(
                  &mut sta_prob_mats.upp_mat_pair_4_el.1[long_v],
                  insert_prob,
                  is_viterbi,
                );
              }
              if trains_score_params {
                sumormax(
                  &mut expected_feature_count_sets.opening_gap_count,
                  insert_prob,
                  is_viterbi,
                );
                sumormax(
                  &mut expected_feature_count_sets.external_loop_accessible_baseunpairing_count,
                  insert_prob,
                  is_viterbi,
                );
              }
              let insert_prob = feature_score_sets.extending_gap_count + part_funcs.part_func_4_insert_2 + backward_term_4_insert_2 - global_part_func + feature_score_sets.external_loop_accessible_baseunpairing_count;
              if produces_access_probs {
                sumormax(
                  &mut sta_prob_mats.upp_mat_pair_4_el.1[long_v],
                  insert_prob,
                  is_viterbi,
                );
              }
              if trains_score_params {
                sumormax(
                  &mut expected_feature_count_sets.extending_gap_count,
                  insert_prob,
                  is_viterbi,
                );
                sumormax(
                  &mut expected_feature_count_sets.external_loop_accessible_baseunpairing_count,
                  insert_prob,
                  is_viterbi,
                );
              }
            },
            None => {},
          }
        }
        if !is_min_gap_ok(&pos_pair, &pseudo_pos_quadruple, max_gap_num_4_il) {
          continue;
        }
        if !(u > T::zero() && v > T::zero()) {
          continue;
        }
        let loop_align_score = feature_score_sets.loop_align_count_mat[base][base_2];
        for i in range(T::one(), u) {
          for j in range_inclusive(u + T::one(), seq_len_pair.0 - T::from_usize(2).unwrap()) {
            let (long_i, long_j) = (i.to_usize().unwrap(), j.to_usize().unwrap());
            let base_pair = (seq_pair.0[long_i], seq_pair.0[long_j]);
            if !is_canonical(&base_pair) {
              continue;
            }
            if !bpp_mat_pair.0.contains_key(&(i, j)) {
              continue;
            }
            let hairpin_loop_score = get_consprob_hairpin_loop_score(feature_score_sets, seq_pair.0, &(long_i, long_j));
            let multi_loop_closing_basepairing_score = get_consprob_multi_loop_closing_basepairing_score(
              feature_score_sets,
              seq_pair.0,
              &(long_i, long_j),
            );
            for k in range(T::one(), v) {
              if !is_min_gap_ok(&(i, k), &pseudo_pos_quadruple, max_gap_num_4_il) {
                continue;
              }
              for l in range(v + T::one(), seq_len_pair.1 - T::one()) {
                let (long_k, long_l) = (k.to_usize().unwrap(), l.to_usize().unwrap());
                if !is_min_gap_ok(&(j, l), &pseudo_pos_quadruple, max_gap_num_4_il) {
                  continue;
                }
                let base_pair_2 = (seq_pair.1[long_k], seq_pair.1[long_l]);
                if !is_canonical(&base_pair_2) {
                  continue;
                }
                if !bpp_mat_pair.1.contains_key(&(k, l)) {
                  continue;
                }
                let hairpin_loop_score_2 = get_consprob_hairpin_loop_score(feature_score_sets, seq_pair.1, &(long_k, long_l));
                let multi_loop_closing_basepairing_score_2 = get_consprob_multi_loop_closing_basepairing_score(
                  feature_score_sets,
                  seq_pair.1,
                  &(long_k, long_l),
                );
                let pos_quadruple = (i, j, k, l);
                match sta_outside_part_func_4d_mat_4_bpas.get(&pos_quadruple) {
                  Some(&part_func_4_bpa) => {
                    let basepair_align_score = feature_score_sets.basepair_align_count_mat[base_pair.0][base_pair.1][base_pair_2.0][base_pair_2.1];
                    let prob_coeff = part_func_4_bpa - global_part_func + basepair_align_score;
                    let ref forward_tmp_part_func_set_mat =
                      sta_part_func_mats.forward_tmp_part_func_set_mats_with_pos_pairs[&(i, k)];
                    let ref backward_tmp_part_func_set_mat_4_decode =
                      sta_part_func_mats.backward_tmp_part_func_set_mats_with_pos_pairs_decode[&(j, l)];
                    let ref forward_tmp_part_func_set_mat_4_2loop =
                      sta_part_func_mats.forward_2loop_tmp_part_func_set_mats_with_pos_quadruples[&pos_quadruple];
                    let ref backward_tmp_part_func_set_mat_4_2loop_decode =
                      sta_part_func_mats.backward_2loop_tmp_part_func_set_mats_with_pos_quadruples_decode[&pos_quadruple];
                    let mut backward_term_4_align_on_sa = NEG_INFINITY;
                    let mut backward_term_4_insert_on_sa = backward_term_4_align_on_sa;
                    let mut backward_term_4_insert_on_sa_2 = backward_term_4_align_on_sa;
                    let mut backward_term_4_align_4_ml = backward_term_4_align_on_sa;
                    let mut backward_term_4_insert_4_ml = backward_term_4_align_on_sa;
                    let mut backward_term_4_insert_4_ml_2 = backward_term_4_align_on_sa;
                    let mut backward_term_4_align_4_bpas_on_mls = backward_term_4_align_on_sa;
                    let mut backward_term_4_insert_4_bpas_on_mls = backward_term_4_align_on_sa;
                    let mut backward_term_4_insert_4_bpas_on_mls_2 = backward_term_4_align_on_sa;
                    let mut backward_term_4_align_on_mls = backward_term_4_align_on_sa;
                    let mut backward_term_4_insert_on_mls = backward_term_4_align_on_sa;
                    let mut backward_term_4_insert_on_mls_2 = backward_term_4_align_on_sa;
                    let mut backward_term_4_align_4_2loop = backward_term_4_align_on_sa;
                    let mut backward_term_4_insert_4_2loop = backward_term_4_align_on_sa;
                    let mut backward_term_4_insert_4_2loop_2 = backward_term_4_align_on_sa;
                    match backward_tmp_part_func_set_mat_4_decode.get(&pos_pair) {
                      Some(part_func_sets) => {
                        backward_term_4_align_on_sa = part_func_sets.part_funcs_on_sa.part_func_4_align;
                        backward_term_4_align_4_ml = part_func_sets.part_funcs_4_ml.part_func_4_align;
                        backward_term_4_align_4_bpas_on_mls = part_func_sets.part_funcs_4_bpas_on_mls.part_func_4_align;
                        backward_term_4_align_on_mls = part_func_sets.part_funcs_on_mls.part_func_4_align;
                        match backward_tmp_part_func_set_mat_4_2loop_decode.get(&pos_pair) {
                          Some(part_funcs_4_2loop) => {
                            backward_term_4_align_4_2loop = part_funcs_4_2loop.part_func_4_align;
                            backward_term_4_insert_4_2loop = part_funcs_4_2loop.part_func_4_insert;
                            backward_term_4_insert_4_2loop_2 = part_funcs_4_2loop.part_func_4_insert_2;
                          }, None => {},
                        }
                        backward_term_4_insert_on_sa = part_func_sets.part_funcs_on_sa.part_func_4_insert;
                        backward_term_4_insert_4_ml = part_func_sets.part_funcs_4_ml.part_func_4_insert;
                        backward_term_4_insert_4_bpas_on_mls = part_func_sets.part_funcs_4_bpas_on_mls.part_func_4_insert;
                        backward_term_4_insert_on_mls = part_func_sets.part_funcs_on_mls.part_func_4_insert;
                        backward_term_4_insert_on_sa_2 = part_func_sets.part_funcs_on_sa.part_func_4_insert_2;
                        backward_term_4_insert_4_ml_2 = part_func_sets.part_funcs_4_ml.part_func_4_insert_2;
                        backward_term_4_insert_4_bpas_on_mls_2 = part_func_sets.part_funcs_4_bpas_on_mls.part_func_4_insert_2;
                        backward_term_4_insert_on_mls_2 = part_func_sets.part_funcs_on_mls.part_func_4_insert_2;
                      }
                      None => {}
                    }
                    let prob_coeff_4_hl = prob_coeff + hairpin_loop_score + hairpin_loop_score_2;
                    let prob_coeff_4_ml = prob_coeff
                      + 2. * feature_score_sets.multi_loop_base_count
                      + multi_loop_closing_basepairing_score
                      + multi_loop_closing_basepairing_score_2;
                    match forward_tmp_part_func_set_mat.get(&pos_pair_4_loop_align) {
                      Some(part_func_sets) => {
                        let loop_align_prob_4_hairpin_loop =
                          prob_coeff_4_hl + loop_align_score + part_func_sets.part_funcs_on_sa.part_func + backward_term_4_align_on_sa;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_hl.0[long_u],
                            loop_align_prob_4_hairpin_loop,
                            is_viterbi,
                          );
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_hl.1[long_v],
                            loop_align_prob_4_hairpin_loop,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1],
                            loop_align_prob_4_hairpin_loop,
                            is_viterbi,
                          );
                        }
                        let loop_align_prob_4_multi_loop =
                          prob_coeff_4_ml + loop_align_score + part_func_sets.part_funcs_on_sa_4_ml.part_func + backward_term_4_align_4_ml + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u],
                            loop_align_prob_4_multi_loop,
                            is_viterbi,
                          );
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v],
                            loop_align_prob_4_multi_loop,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1],
                            loop_align_prob_4_multi_loop,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            (2. as Prob).ln() + loop_align_prob_4_multi_loop,
                            is_viterbi,
                          );
                        }
                        let loop_align_prob_4_multi_loop = prob_coeff_4_ml
                          + loop_align_score
                          + part_func_sets.part_funcs_4_first_bpas_on_mls.part_func
                          + backward_term_4_align_4_bpas_on_mls + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u],
                            loop_align_prob_4_multi_loop,
                            is_viterbi,
                          );
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v],
                            loop_align_prob_4_multi_loop,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1],
                            loop_align_prob_4_multi_loop,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            (2. as Prob).ln() + loop_align_prob_4_multi_loop,
                            is_viterbi,
                          );
                        }
                        let loop_align_prob_4_multi_loop =
                          prob_coeff_4_ml + loop_align_score + part_func_sets.part_funcs_4_ml.part_func + backward_term_4_align_on_mls + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u],
                            loop_align_prob_4_multi_loop,
                            is_viterbi,
                          );
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v],
                            loop_align_prob_4_multi_loop,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1],
                            loop_align_prob_4_multi_loop,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            (2. as Prob).ln() + loop_align_prob_4_multi_loop,
                            is_viterbi,
                          );
                        }
                        match forward_tmp_part_func_set_mat_4_2loop.get(&pos_pair_4_loop_align) {
                          Some(part_funcs_4_2loop) => {
                            let loop_align_prob_4_2loop =
                              prob_coeff + loop_align_score + part_funcs_4_2loop.part_func + backward_term_4_align_on_sa;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.0[long_u],
                                loop_align_prob_4_2loop,
                                is_viterbi,
                              );
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.1[long_v],
                                loop_align_prob_4_2loop,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1],
                                loop_align_prob_4_2loop,
                                is_viterbi,
                              );
                            }
                            let loop_align_prob_4_2loop =
                              prob_coeff + loop_align_score + part_func_sets.part_funcs_on_sa.part_func + backward_term_4_align_4_2loop;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.0[long_u],
                                loop_align_prob_4_2loop,
                                is_viterbi,
                              );
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.1[long_v],
                                loop_align_prob_4_2loop,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1],
                                loop_align_prob_4_2loop,
                                is_viterbi,
                              );
                            }
                          }, None => {},
                        }
                      }
                      None => {}
                    }
                    match forward_tmp_part_func_set_mat.get(&pos_pair_4_insert) {
                      Some(part_func_sets) => {
                        let upp_4_hl = prob_coeff_4_hl
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_on_sa.part_func_4_align
                          + backward_term_4_insert_on_sa;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_hl.0[long_u],
                            upp_4_hl,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_hl,
                            is_viterbi,
                          );
                        }
                        let upp_4_hl = prob_coeff_4_hl
                          + feature_score_sets.extending_gap_count
                          + part_func_sets.part_funcs_on_sa.part_func_4_insert
                          + backward_term_4_insert_on_sa;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_hl.0[long_u],
                            upp_4_hl,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.extending_gap_count,
                            upp_4_hl,
                            is_viterbi,
                          );
                        }
                        let upp_4_hl = prob_coeff_4_hl
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_on_sa.part_func_4_insert_2
                          + backward_term_4_insert_on_sa;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_hl.0[long_u],
                            upp_4_hl,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_hl,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_on_sa_4_ml.part_func_4_align
                          + backward_term_4_insert_4_ml + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.extending_gap_count
                          + part_func_sets.part_funcs_on_sa_4_ml.part_func_4_insert
                          + backward_term_4_insert_4_ml + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.extending_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_on_sa_4_ml.part_func_4_insert_2
                          + backward_term_4_insert_4_ml + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_4_first_bpas_on_mls.part_func_4_align
                          + backward_term_4_insert_4_bpas_on_mls + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.extending_gap_count
                          + part_func_sets.part_funcs_4_first_bpas_on_mls.part_func_4_insert
                          + backward_term_4_insert_4_bpas_on_mls + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.extending_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_4_first_bpas_on_mls.part_func_4_insert_2
                          + backward_term_4_insert_4_bpas_on_mls + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_4_ml.part_func_4_align
                          + backward_term_4_insert_on_mls + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.extending_gap_count
                          + part_func_sets.part_funcs_4_ml.part_func_4_insert
                          + backward_term_4_insert_on_mls + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.extending_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_4_ml.part_func_4_insert_2
                          + backward_term_4_insert_on_mls + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        match forward_tmp_part_func_set_mat_4_2loop.get(&pos_pair_4_insert) {
                          Some(part_funcs_4_2loop) => {
                            let upp_4_2l = prob_coeff + feature_score_sets.opening_gap_count
                              + part_funcs_4_2loop.part_func_4_align
                              + backward_term_4_insert_on_sa;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.0[long_u],
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.opening_gap_count,
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            let upp_4_2l = prob_coeff + feature_score_sets.extending_gap_count
                              + part_funcs_4_2loop.part_func_4_insert
                              + backward_term_4_insert_on_sa;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.0[long_u],
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.extending_gap_count,
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            let upp_4_2l = prob_coeff + feature_score_sets.opening_gap_count
                              + part_funcs_4_2loop.part_func_4_insert_2
                              + backward_term_4_insert_on_sa;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.0[long_u],
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.opening_gap_count,
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            let upp_4_2l = prob_coeff + feature_score_sets.opening_gap_count
                              + part_func_sets.part_funcs_on_sa.part_func_4_align
                              + backward_term_4_insert_4_2loop;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.0[long_u],
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.opening_gap_count,
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            let upp_4_2l = prob_coeff + feature_score_sets.extending_gap_count
                              + part_func_sets.part_funcs_on_sa.part_func_4_insert
                              + backward_term_4_insert_4_2loop;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.0[long_u],
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.extending_gap_count,
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            let upp_4_2l = prob_coeff + feature_score_sets.opening_gap_count
                              + part_func_sets.part_funcs_on_sa.part_func_4_insert_2
                              + backward_term_4_insert_4_2loop;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.0[long_u],
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.opening_gap_count,
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                          }, None => {},
                        }
                      }
                      None => {}
                    }
                    match forward_tmp_part_func_set_mat.get(&pos_pair_4_insert_2) {
                      Some(part_func_sets) => {
                        let upp_4_hl = prob_coeff_4_hl
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_on_sa.part_func_4_align
                          + backward_term_4_insert_on_sa_2;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_hl.1[long_v],
                            upp_4_hl,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_hl,
                            is_viterbi,
                          );
                        }
                        let upp_4_hl = prob_coeff_4_hl
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_on_sa.part_func_4_insert
                          + backward_term_4_insert_on_sa_2;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_hl.1[long_v],
                            upp_4_hl,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_hl,
                            is_viterbi,
                          );
                        }
                        let upp_4_hl = prob_coeff_4_hl
                          + feature_score_sets.extending_gap_count
                          + part_func_sets.part_funcs_on_sa.part_func_4_insert_2
                          + backward_term_4_insert_on_sa_2;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_hl.1[long_v],
                            upp_4_hl,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.extending_gap_count,
                            upp_4_hl,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_on_sa_4_ml.part_func_4_align
                          + backward_term_4_insert_4_ml_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_on_sa_4_ml.part_func_4_insert
                          + backward_term_4_insert_4_ml_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.extending_gap_count
                          + part_func_sets.part_funcs_on_sa_4_ml.part_func_4_insert_2
                          + backward_term_4_insert_4_ml_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.extending_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_4_first_bpas_on_mls.part_func_4_align
                          + backward_term_4_insert_4_bpas_on_mls_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_4_first_bpas_on_mls.part_func_4_insert
                          + backward_term_4_insert_4_bpas_on_mls_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.extending_gap_count
                          + part_func_sets.part_funcs_4_first_bpas_on_mls.part_func_4_insert_2
                          + backward_term_4_insert_4_bpas_on_mls_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.extending_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_4_ml.part_func_4_align
                          + backward_term_4_insert_on_mls_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.opening_gap_count
                          + part_func_sets.part_funcs_4_ml.part_func_4_insert
                          + backward_term_4_insert_on_mls_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.opening_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        let upp_4_ml = prob_coeff_4_ml
                          + feature_score_sets.extending_gap_count
                          + part_func_sets.part_funcs_4_ml.part_func_4_insert_2
                          + backward_term_4_insert_on_mls_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                        if produces_access_probs {
                          sumormax(
                            &mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v],
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        if trains_score_params {
                          sumormax(
                            &mut expected_feature_count_sets.extending_gap_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                          sumormax(
                            &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                            upp_4_ml,
                            is_viterbi,
                          );
                        }
                        match forward_tmp_part_func_set_mat_4_2loop.get(&pos_pair_4_insert_2) {
                          Some(part_funcs_4_2loop) => {
                            let upp_4_2l = prob_coeff + feature_score_sets.opening_gap_count
                              + part_funcs_4_2loop.part_func_4_align
                              + backward_term_4_insert_on_sa_2;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.1[long_v],
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.opening_gap_count,
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            let upp_4_2l = prob_coeff + feature_score_sets.opening_gap_count
                              + part_funcs_4_2loop.part_func_4_insert
                              + backward_term_4_insert_on_sa_2;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.1[long_v],
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.opening_gap_count,
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            let upp_4_2l = prob_coeff + feature_score_sets.extending_gap_count
                              + part_funcs_4_2loop.part_func_4_insert_2
                              + backward_term_4_insert_on_sa_2;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.1[long_v],
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.extending_gap_count,
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            let upp_4_2l = prob_coeff + feature_score_sets.opening_gap_count
                              + part_func_sets.part_funcs_on_sa.part_func_4_align
                              + backward_term_4_insert_4_2loop_2;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.1[long_v],
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.opening_gap_count,
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            let upp_4_2l = prob_coeff + feature_score_sets.opening_gap_count
                              + part_func_sets.part_funcs_on_sa.part_func_4_insert
                              + backward_term_4_insert_4_2loop_2;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.1[long_v],
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.opening_gap_count,
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            let upp_4_2l = prob_coeff + feature_score_sets.extending_gap_count
                              + part_func_sets.part_funcs_on_sa.part_func_4_insert_2
                              + backward_term_4_insert_4_2loop_2;
                            if produces_access_probs {
                              sumormax(
                                &mut sta_prob_mats.upp_mat_pair_4_2l.1[long_v],
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                            if trains_score_params {
                              sumormax(
                                &mut expected_feature_count_sets.extending_gap_count,
                                upp_4_2l,
                                is_viterbi,
                              );
                            }
                          }, None => {},
                        }
                      }
                      None => {}
                    }
                  }
                  None => {}
                }
              }
            }
          }
        }
      }
    }
    if produces_access_probs {
      for bpp in sta_prob_mats.access_bpp_mat_pair_4_2l.0.values_mut() {
        *bpp = bpp.exp();
      }
      for bpp in sta_prob_mats.access_bpp_mat_pair_4_2l.1.values_mut() {
        *bpp = bpp.exp();
      }
      for bpp in sta_prob_mats.access_bpp_mat_pair_4_ml.0.values_mut() {
        *bpp = bpp.exp();
      }
      for bpp in sta_prob_mats.access_bpp_mat_pair_4_ml.1.values_mut() {
        *bpp = bpp.exp();
      }
      for bpp in sta_prob_mats.bpp_mat_pair_4_el.0.values_mut() {
        *bpp = bpp.exp();
      }
      for bpp in sta_prob_mats.bpp_mat_pair_4_el.1.values_mut() {
        *bpp = bpp.exp();
      }
      for upp in sta_prob_mats.upp_mat_pair_4_hl.0.iter_mut() {
        *upp = upp.exp();
      }
      for upp in sta_prob_mats.upp_mat_pair_4_hl.1.iter_mut() {
        *upp = upp.exp();
      }
      for upp in sta_prob_mats.upp_mat_pair_4_ml.0.iter_mut() {
        *upp = upp.exp();
      }
      for upp in sta_prob_mats.upp_mat_pair_4_ml.1.iter_mut() {
        *upp = upp.exp();
      }
      for upp in sta_prob_mats.upp_mat_pair_4_2l.0.iter_mut() {
        *upp = upp.exp();
      }
      for upp in sta_prob_mats.upp_mat_pair_4_2l.1.iter_mut() {
        *upp = upp.exp();
      }
      for upp in sta_prob_mats.upp_mat_pair_4_el.0.iter_mut() {
        *upp = upp.exp();
      }
      for upp in sta_prob_mats.upp_mat_pair_4_el.1.iter_mut() {
        *upp = upp.exp();
      }
    }
    if trains_score_params {
      for count in expected_feature_count_sets.hairpin_loop_length_counts.iter_mut() {
        *count = count.exp();
      }
      for count in expected_feature_count_sets.bulge_loop_length_counts.iter_mut() {
        *count = count.exp();
      }
      for count in expected_feature_count_sets.interior_loop_length_counts.iter_mut() {
        *count = count.exp();
      }
      for count in expected_feature_count_sets.interior_loop_length_counts_symm.iter_mut() {
        *count = count.exp();
      }
      for count in expected_feature_count_sets.interior_loop_length_counts_asymm.iter_mut() {
        *count = count.exp();
      }
      for count_3d_mat in expected_feature_count_sets.stack_count_mat.iter_mut() {
        for count_2d_mat in count_3d_mat.iter_mut() {
          for counts in count_2d_mat.iter_mut() {
            for count in counts.iter_mut() {
              *count = count.exp();
            }
          }
        }
      }
      for count_3d_mat in expected_feature_count_sets.terminal_mismatch_count_mat.iter_mut() {
        for count_2d_mat in count_3d_mat.iter_mut() {
          for counts in count_2d_mat.iter_mut() {
            for count in counts.iter_mut() {
              *count = count.exp();
            }
          }
        }
      }
      for count_2d_mat in expected_feature_count_sets.left_dangle_count_mat.iter_mut() {
        for counts in count_2d_mat.iter_mut() {
          for count in counts.iter_mut() {
            *count = count.exp();
          }
        }
      }
      for count_2d_mat in expected_feature_count_sets.right_dangle_count_mat.iter_mut() {
        for counts in count_2d_mat.iter_mut() {
          for count in counts.iter_mut() {
            *count = count.exp();
          }
        }
      }
      for counts in expected_feature_count_sets.interior_loop_length_count_mat_explicit.iter_mut() {
        for count in counts.iter_mut() {
          *count = count.exp();
        }
      }
      for count in expected_feature_count_sets.bulge_loop_0x1_length_counts.iter_mut() {
        *count = count.exp();
      }
      for counts in expected_feature_count_sets.interior_loop_1x1_length_count_mat.iter_mut() {
        for count in counts.iter_mut() {
          *count = count.exp();
        }
      }
      for counts in expected_feature_count_sets.helix_end_count_mat.iter_mut() {
        for count in counts.iter_mut() {
          *count = count.exp();
        }
      }
      for counts in expected_feature_count_sets.base_pair_count_mat.iter_mut() {
        for count in counts.iter_mut() {
          *count = count.exp();
        }
      }
      expected_feature_count_sets.multi_loop_base_count = expected_feature_count_sets.multi_loop_base_count.exp();
      expected_feature_count_sets.multi_loop_basepairing_count = expected_feature_count_sets.multi_loop_basepairing_count.exp();
      expected_feature_count_sets.multi_loop_accessible_baseunpairing_count = expected_feature_count_sets.multi_loop_accessible_baseunpairing_count.exp();
      expected_feature_count_sets.external_loop_accessible_basepairing_count = expected_feature_count_sets.external_loop_accessible_basepairing_count.exp();
      expected_feature_count_sets.external_loop_accessible_baseunpairing_count = expected_feature_count_sets.external_loop_accessible_baseunpairing_count.exp();
      for count_3d_mat in expected_feature_count_sets.basepair_align_count_mat.iter_mut() {
        for count_2d_mat in count_3d_mat.iter_mut() {
          for counts in count_2d_mat.iter_mut() {
            for count in counts.iter_mut() {
              *count = count.exp();
            }
          }
        }
      }
      for counts in expected_feature_count_sets.loop_align_count_mat.iter_mut() {
        for count in counts.iter_mut() {
          *count = count.exp();
        }
      }
      expected_feature_count_sets.opening_gap_count = expected_feature_count_sets.opening_gap_count.exp();
      expected_feature_count_sets.extending_gap_count = expected_feature_count_sets.extending_gap_count.exp();
    }
  }
  sta_prob_mats
}

pub fn pct_of_prob_mats<T>(
  prob_mats_with_rna_id_pairs: &StaProbMatsWithRnaIdPairs<T>,
  rna_id: RnaId,
  num_of_rnas: usize,
  bpp_mat: &SparseProbMat<T>,
  upp_mat_len: usize,
  produces_access_probs: bool,
) -> PctStaProbMats<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord,
{
  let weight = 1. / (num_of_rnas - 1) as Prob;
  let mut pct_prob_mats = PctStaProbMats::new(upp_mat_len);
  pct_prob_mats.bpp_mat_on_ss = bpp_mat.clone();
  for rna_id_2 in 0..num_of_rnas {
    if rna_id == rna_id_2 {
      continue;
    }
    let rna_id_pair = if rna_id < rna_id_2 {
      (rna_id, rna_id_2)
    } else {
      (rna_id_2, rna_id)
    };
    let ref ref_2_prob_mats = prob_mats_with_rna_id_pairs[&rna_id_pair];
    let ref_2_bpp_mat = if rna_id < rna_id_2 {
      &ref_2_prob_mats.bpp_mat_pair.0
    } else {
      &ref_2_prob_mats.bpp_mat_pair.1
    };
    for (pos_pair, &bpp) in ref_2_bpp_mat.iter() {
      let weighted_bpp = weight * bpp;
      match pct_prob_mats.bpp_mat.get_mut(pos_pair) {
        Some(bpp) => {
          *bpp += weighted_bpp;
        }
        None => {
          pct_prob_mats.bpp_mat.insert(*pos_pair, weighted_bpp);
        }
      }
    }
    let ref_2_upp_mat = if rna_id < rna_id_2 {
      &ref_2_prob_mats.upp_mat_pair.0
    } else {
      &ref_2_prob_mats.upp_mat_pair.1
    };
    for (i, &upp) in ref_2_upp_mat.iter().enumerate() {
      let weighted_upp = weight * upp;
      pct_prob_mats.upp_mat[i] += weighted_upp;
    }
    if produces_access_probs {
      let ref_2_access_bpp_mat_4_2l = if rna_id < rna_id_2 {
        &ref_2_prob_mats.access_bpp_mat_pair_4_2l.0
      } else {
        &ref_2_prob_mats.access_bpp_mat_pair_4_2l.1
      };
      for (pos_pair, &bpp) in ref_2_access_bpp_mat_4_2l.iter() {
        let weighted_bpp = weight * bpp;
        match pct_prob_mats.access_bpp_mat_4_2l.get_mut(pos_pair) {
          Some(bpp_4_2l) => {
            *bpp_4_2l += weighted_bpp;
          }
          None => {
            pct_prob_mats
              .access_bpp_mat_4_2l
              .insert(*pos_pair, weighted_bpp);
          }
        }
      }
      let ref_2_access_bpp_mat_4_ml = if rna_id < rna_id_2 {
        &ref_2_prob_mats.access_bpp_mat_pair_4_ml.0
      } else {
        &ref_2_prob_mats.access_bpp_mat_pair_4_ml.1
      };
      for (pos_pair, &bpp) in ref_2_access_bpp_mat_4_ml.iter() {
        let weighted_bpp = weight * bpp;
        match pct_prob_mats.access_bpp_mat_4_ml.get_mut(pos_pair) {
          Some(bpp_4_ml) => {
            *bpp_4_ml += weighted_bpp;
          }
          None => {
            pct_prob_mats
              .access_bpp_mat_4_ml
              .insert(*pos_pair, weighted_bpp);
          }
        }
      }
      let ref_2_bpp_mat_4_el = if rna_id < rna_id_2 {
        &ref_2_prob_mats.bpp_mat_pair_4_el.0
      } else {
        &ref_2_prob_mats.bpp_mat_pair_4_el.1
      };
      for (pos_pair, &bpp) in ref_2_bpp_mat_4_el.iter() {
        let weighted_bpp = weight * bpp;
        match pct_prob_mats.bpp_mat_4_el.get_mut(pos_pair) {
          Some(bpp_4_el) => {
            *bpp_4_el += weighted_bpp;
          }
          None => {
            pct_prob_mats.bpp_mat_4_el.insert(*pos_pair, weighted_bpp);
          }
        }
      }
      let ref_2_upp_mat = if rna_id < rna_id_2 {
        &ref_2_prob_mats.upp_mat_pair_4_hl.0
      } else {
        &ref_2_prob_mats.upp_mat_pair_4_hl.1
      };
      for (i, &upp) in ref_2_upp_mat.iter().enumerate() {
        let weighted_upp = weight * upp;
        pct_prob_mats.upp_mat_4_hl[i] += weighted_upp;
      }
      let ref_2_upp_mat = if rna_id < rna_id_2 {
        &ref_2_prob_mats.upp_mat_pair_4_2l.0
      } else {
        &ref_2_prob_mats.upp_mat_pair_4_2l.1
      };
      for (i, &upp) in ref_2_upp_mat.iter().enumerate() {
        let weighted_upp = weight * upp;
        pct_prob_mats.upp_mat_4_2l[i] += weighted_upp;
      }
      let ref_2_upp_mat = if rna_id < rna_id_2 {
        &ref_2_prob_mats.upp_mat_pair_4_ml.0
      } else {
        &ref_2_prob_mats.upp_mat_pair_4_ml.1
      };
      for (i, &upp) in ref_2_upp_mat.iter().enumerate() {
        let weighted_upp = weight * upp;
        pct_prob_mats.upp_mat_4_ml[i] += weighted_upp;
      }
      let ref_2_upp_mat = if rna_id < rna_id_2 {
        &ref_2_prob_mats.upp_mat_pair_4_el.0
      } else {
        &ref_2_prob_mats.upp_mat_pair_4_el.1
      };
      for (i, &upp) in ref_2_upp_mat.iter().enumerate() {
        let weighted_upp = weight * upp;
        pct_prob_mats.upp_mat_4_el[i] += weighted_upp;
      }
    }
  }
  if produces_access_probs {
    for (i, &upp) in pct_prob_mats.upp_mat_4_hl.iter().enumerate() {
      pct_prob_mats.max_upp_mat[i] = upp;
    }
    for (i, &upp) in pct_prob_mats.upp_mat_4_2l.iter().enumerate() {
      let old_upp = pct_prob_mats.max_upp_mat[i];
      if upp > old_upp {
        pct_prob_mats.max_upp_mat[i] = upp;
      }
    }
    for (i, &upp) in pct_prob_mats.upp_mat_4_ml.iter().enumerate() {
      let old_upp = pct_prob_mats.max_upp_mat[i];
      if upp > old_upp {
        pct_prob_mats.max_upp_mat[i] = upp;
      }
    }
    for (i, &upp) in pct_prob_mats.upp_mat_4_el.iter().enumerate() {
      let old_upp = pct_prob_mats.max_upp_mat[i];
      if upp > old_upp {
        pct_prob_mats.max_upp_mat[i] = upp;
      }
    }
    for (pos_pair, &bpp) in pct_prob_mats.access_bpp_mat_4_2l.iter() {
      pct_prob_mats.max_bpp_mat.insert(*pos_pair, bpp);
    }
    for (pos_pair, &bpp) in pct_prob_mats.access_bpp_mat_4_ml.iter() {
      match pct_prob_mats.max_bpp_mat.get_mut(pos_pair) {
        Some(old_bpp) => {
          if bpp > *old_bpp {
            *old_bpp = bpp;
          }
        }
        None => {
          pct_prob_mats.max_bpp_mat.insert(*pos_pair, bpp);
        }
      }
    }
    for (pos_pair, &bpp) in pct_prob_mats.bpp_mat_4_el.iter() {
      match pct_prob_mats.max_bpp_mat.get_mut(pos_pair) {
        Some(old_bpp) => {
          if bpp > *old_bpp {
            *old_bpp = bpp;
          }
        }
        None => {
          pct_prob_mats.max_bpp_mat.insert(*pos_pair, bpp);
        }
      }
    }
  }
  pct_prob_mats
}

pub fn get_max_bp_span<T>(sparse_bpp_mat: &SparseProbMat<T>) -> T
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord,
{
  sparse_bpp_mat
    .iter()
    .map(|(pos_pair, _)| pos_pair.1 - pos_pair.0 + T::one())
    .max()
    .unwrap()
}

pub fn print_program_usage(program_name: &str, opts: &Options) {
  let program_usage = format!("The usage of this program: {} [options]", program_name);
  print!("{}", opts.usage(&program_usage));
}

pub fn get_seq_len_diff<T>(pos_quadruple: &PosQuadruple<T>) -> T
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord,
{
  let seq_len_pair = (
    pos_quadruple.1 + T::one() - pos_quadruple.0,
    pos_quadruple.3 + T::one() - pos_quadruple.2,
  );
  max(seq_len_pair.0, seq_len_pair.1) - min(seq_len_pair.0, seq_len_pair.1)
}

pub fn get_diff(x: usize, y: usize) -> usize {
  max(x, y) - min(x, y)
}

pub fn remove_small_bpps_from_bpp_mat<T>(
  sparse_bpp_mat: &SparseProbMat<T>,
  min_bpp: Prob,
) -> SparseProbMat<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord,
{
  sparse_bpp_mat
    .iter()
    .filter(|(_, &bpp)| bpp >= min_bpp)
    .map(|(&(i, j), &bpp)| ((i + T::one(), j + T::one()), bpp))
    .collect()
}

fn is_min_gap_ok<T>(pos_pair: &PosPair<T>, pos_quadruple: &PosQuadruple<T>, max_gap_num: T) -> bool
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord,
{
  let min_gap_num_1 =
    get_seq_len_diff::<T>(&(pos_quadruple.0, pos_pair.0, pos_quadruple.2, pos_pair.1));
  let min_gap_num_2 =
    get_seq_len_diff::<T>(&(pos_pair.0, pos_quadruple.1, pos_pair.1, pos_quadruple.3));
  if min_gap_num_1 <= max_gap_num && min_gap_num_2 <= max_gap_num {
    true
  } else {
    false
  }
}

pub fn sumormax(sum: &mut FreeEnergy, new_term: FreeEnergy, is_viterbi: bool) {
  if new_term > NEG_INFINITY {
    *sum = if !sum.is_finite() {
      new_term
    } else {
      let max = sum.max(new_term);
      if is_viterbi {
        max
      } else {
        max
          + ((if *sum == max {
            new_term - max
          } else {
            *sum - max
          })
          .exp()
            + 1.)
            .ln()
      }
    };
  }
}

pub fn get_consprob_multi_loop_closing_basepairing_score(
  feature_score_sets: &FeatureCountSets,
  seq: SeqSlice,
  pos_pair: &(usize, usize),
) -> FeatureCount {
  let base_pair = (seq[pos_pair.0], seq[pos_pair.1]);
  let mismatch_pair = (seq[pos_pair.0 + 1], seq[pos_pair.1 - 1]);
  feature_score_sets.multi_loop_base_count
    + feature_score_sets.multi_loop_basepairing_count
    + feature_score_sets.left_dangle_count_mat[base_pair.0][base_pair.1]
      [mismatch_pair.0]
    + feature_score_sets.right_dangle_count_mat[base_pair.0][base_pair.1]
      [mismatch_pair.1]
    + feature_score_sets.helix_end_count_mat[base_pair.0][base_pair.1]
}

pub fn get_consprob_multi_loop_accessible_basepairing_score(
  feature_score_sets: &FeatureCountSets,
  seq: SeqSlice,
  pos_pair: &(usize, usize),
) -> FeatureCount {
  let base_pair = (seq[pos_pair.1], seq[pos_pair.0]);
  let mismatch_pair = (seq[pos_pair.1 + 1], seq[pos_pair.0 - 1]);
  if mismatch_pair.0 == PSEUDO_BASE || mismatch_pair.1 == PSEUDO_BASE {
    NEG_INFINITY
  } else {
    feature_score_sets.multi_loop_basepairing_count
      + feature_score_sets.left_dangle_count_mat[base_pair.0][base_pair.1]
        [mismatch_pair.0]
      + feature_score_sets.right_dangle_count_mat[base_pair.0][base_pair.1]
        [mismatch_pair.1]
      + feature_score_sets.helix_end_count_mat[base_pair.0][base_pair.1]
      + feature_score_sets.base_pair_count_mat[base_pair.0][base_pair.1]
  }
}

pub fn get_consprob_external_loop_accessible_basepairing_score(
  feature_score_sets: &FeatureCountSets,
  seq: SeqSlice,
  pos_pair: &(usize, usize),
) -> FeatureCount {
  let seq_len = seq.len();
  let base_pair = (seq[pos_pair.1], seq[pos_pair.0]);
  let mismatch_pair = (seq[pos_pair.1 + 1], seq[pos_pair.0 - 1]);
  let left_terminal_mismatch_score = if pos_pair.1 < seq_len - 2 {
    feature_score_sets.left_dangle_count_mat[base_pair.0][base_pair.1]
      [mismatch_pair.0]
  } else {
    0.
  };
  let right_terminal_mismatch_score = if pos_pair.0 > 1 {
    feature_score_sets.right_dangle_count_mat[base_pair.0][base_pair.1]
      [mismatch_pair.1]
  } else {
    0.
  };
  left_terminal_mismatch_score
    + right_terminal_mismatch_score
    + feature_score_sets.external_loop_accessible_basepairing_count
    + feature_score_sets.helix_end_count_mat[base_pair.0][base_pair.1]
    + feature_score_sets.base_pair_count_mat[base_pair.0][base_pair.1]
}

pub fn get_consprob_hairpin_loop_score(
  feature_score_sets: &FeatureCountSets,
  seq: SeqSlice,
  pos_pair: &(usize, usize),
) -> FeatureCount {
  let hairpin_loop_len = pos_pair.1 - pos_pair.0 - 1;
  if hairpin_loop_len > CONSPROB_MAX_HAIRPIN_LOOP_LEN {
    NEG_INFINITY
  } else {
    let base_pair = (seq[pos_pair.0], seq[pos_pair.1]);
    let mismatch_pair = (seq[pos_pair.0 + 1], seq[pos_pair.1 - 1]);
    feature_score_sets.hairpin_loop_length_counts_cumulative[hairpin_loop_len - CONSPROB_MIN_HAIRPIN_LOOP_LEN]
      + feature_score_sets.terminal_mismatch_count_mat[base_pair.0][base_pair.1]
        [mismatch_pair.0][mismatch_pair.1]
      + feature_score_sets.helix_end_count_mat[base_pair.0][base_pair.1]
  }
}

pub fn get_consprob_twoloop_score(
  feature_score_sets: &FeatureCountSets,
  seq: SeqSlice,
  pos_pair_closing_loop: &(usize, usize),
  pos_pair_accessible: &(usize, usize),
) -> FeatureCount {
  let is_stack = pos_pair_closing_loop.0 + 1 == pos_pair_accessible.0
    && pos_pair_closing_loop.1 - 1 == pos_pair_accessible.1;
  let is_bulge_loop = (pos_pair_closing_loop.0 + 1 == pos_pair_accessible.0
    || pos_pair_closing_loop.1 - 1 == pos_pair_accessible.1)
    && !is_stack;
  let base_pair_closing_loop = (seq[pos_pair_closing_loop.0], seq[pos_pair_closing_loop.1]);
  let base_pair_accessible = (seq[pos_pair_accessible.0], seq[pos_pair_accessible.1]);
  let mismatch_pair_1 = (
    seq[pos_pair_closing_loop.0 + 1],
    seq[pos_pair_closing_loop.1 - 1],
  );
  let mismatch_pair_2 = (
    seq[pos_pair_accessible.1 + 1],
    seq[pos_pair_accessible.0 - 1],
  );
  let loop_len_pair = (
    pos_pair_accessible.0 - pos_pair_closing_loop.0 - 1,
    pos_pair_closing_loop.1 - pos_pair_accessible.1 - 1,
  );
   feature_score_sets.base_pair_count_mat[base_pair_accessible.0][base_pair_accessible.1] + if is_stack {
    feature_score_sets.stack_count_mat[base_pair_closing_loop.0][base_pair_closing_loop.1]
      [base_pair_accessible.0][base_pair_accessible.1]
  } else {
    let helix_end_score_sum = feature_score_sets.helix_end_count_mat[base_pair_closing_loop.0]
      [base_pair_closing_loop.1]
      + feature_score_sets.helix_end_count_mat[base_pair_accessible.1][base_pair_accessible.0];
    helix_end_score_sum
      + feature_score_sets.terminal_mismatch_count_mat[base_pair_closing_loop.0][base_pair_closing_loop.1][mismatch_pair_1.0][mismatch_pair_1.1]
      + feature_score_sets.terminal_mismatch_count_mat[base_pair_accessible.1][base_pair_accessible.0][mismatch_pair_2.0][mismatch_pair_2.1] + if is_bulge_loop {
      feature_score_sets.bulge_loop_length_counts_cumulative[loop_len_pair.0 + loop_len_pair.1 - 1]
        + if loop_len_pair.0 + loop_len_pair.1 == 1 {
          feature_score_sets.bulge_loop_0x1_length_counts[if loop_len_pair.0 == 0 {mismatch_pair_1.1} else {mismatch_pair_1.0}]
        } else {0.}
    } else {
      let diff = get_diff(loop_len_pair.0, loop_len_pair.1);
      feature_score_sets.interior_loop_length_counts_cumulative[loop_len_pair.0 + loop_len_pair.1 - 2]
        + if diff == 0 {
          feature_score_sets.interior_loop_length_counts_symm_cumulative[loop_len_pair.0 - 1]
        } else {
          feature_score_sets.interior_loop_length_counts_asymm_cumulative[diff - 1]
        }
        + if loop_len_pair.0 == 1 && loop_len_pair.1 == 1 {
          feature_score_sets.interior_loop_1x1_length_count_mat[mismatch_pair_1.0][mismatch_pair_1.1]
        } else {0.}
        + if loop_len_pair.0 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT && loop_len_pair.1 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT {
          feature_score_sets.interior_loop_length_count_mat_explicit[loop_len_pair.0 - 1][loop_len_pair.1 - 1]
        } else {0.}
    }
  }
}

pub fn get_dict_min_stack(base_pair_closing_loop: &BasePair, base_pair_accessible: &BasePair) -> (BasePair, BasePair) {
  let stack = (*base_pair_closing_loop, *base_pair_accessible);
  let inverse_stack = ((base_pair_accessible.1, base_pair_accessible.0), (base_pair_closing_loop.1, base_pair_closing_loop.0));
  if stack < inverse_stack {
    stack
  } else {
    inverse_stack
  }
}

pub fn get_dict_min_basepair_align(base_pair: &BasePair, base_pair_2: &BasePair) -> (BasePair, BasePair) {
  let basepair_align = (*base_pair, *base_pair_2);
  let inverse_basepair_align = (*base_pair_2, *base_pair);
  if basepair_align < inverse_basepair_align {
    basepair_align
  } else {
    inverse_basepair_align
  }
}

pub fn get_dict_min_loop_align(loop_align: &BasePair) -> BasePair {
  let inverse_loop_align = (loop_align.1, loop_align.0);
  if *loop_align < inverse_loop_align {
    *loop_align
  } else {
    inverse_loop_align
  }
}

pub fn get_dict_min_loop_len_pair(loop_len_pair: &(usize, usize)) -> (usize, usize) {
  let inverse_loop_len_pair = (loop_len_pair.1, loop_len_pair.0);
  if *loop_len_pair < inverse_loop_len_pair {
    *loop_len_pair
  } else {
    inverse_loop_len_pair
  }
}

pub fn get_dict_min_nuc_pair(nuc_pair: &(usize, usize)) -> (usize, usize) {
  get_dict_min_loop_len_pair(nuc_pair)
}

pub fn get_dict_min_mismatch_pair(mismatch_pair: &BasePair) -> BasePair {
  get_dict_min_loop_align(mismatch_pair)
}

pub fn get_dict_min_base_pair(base_pair: &BasePair) -> BasePair {
  get_dict_min_loop_align(base_pair)
}

pub fn get_num_of_multiloop_baseunpairing_nucs(pos_pair_closing_loop: &(usize, usize), pos_pairs_in_loop: &Vec<(usize, usize)>, seq: SeqSlice) -> usize {
  let mut count = 0;
  let mut prev_pos_pair = (0, 0);
  for pos_pair_in_loop in pos_pairs_in_loop {
    let start = if prev_pos_pair == (0, 0) {
      pos_pair_closing_loop.0 + 1
    } else {
      prev_pos_pair.1 + 1
    };
    for i in start .. pos_pair_in_loop.0 {
      count += if seq[i] != PSEUDO_BASE {1} else {0};
    }
    prev_pos_pair = *pos_pair_in_loop;
  }
  for i in prev_pos_pair.1 + 1 .. pos_pair_closing_loop.1 {
    count += if seq[i] != PSEUDO_BASE {1} else {0};
  }
  count
}

pub fn get_num_of_externalloop_baseunpairing_nucs(pos_pairs_in_loop: &Vec<(usize, usize)>, seq: SeqSlice) -> usize {
  let mut count = 0;
  let mut prev_pos_pair = (0, 0);
  for pos_pair_in_loop in pos_pairs_in_loop {
    let start = if prev_pos_pair == (0, 0) {
      0
    } else {
      prev_pos_pair.1 + 1
    };
    for i in start .. pos_pair_in_loop.0 {
      count += if seq[i] != PSEUDO_BASE {1} else {0};
    }
    prev_pos_pair = *pos_pair_in_loop;
  }
  for i in prev_pos_pair.1 + 1 .. seq.len() {
    count += if seq[i] != PSEUDO_BASE {1} else {0};
  }
  count
}

pub fn consprob<T>(
  thread_pool: &mut Pool,
  fasta_records: &FastaRecords,
  min_bpp: Prob,
  offset_4_max_gap_num: T,
  produces_access_probs: bool,
) -> ProbMatSets<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send,
{
  let num_of_fasta_records = fasta_records.len();
  let mut bpp_mats = vec![SparseProbMat::<T>::new(); num_of_fasta_records];
  let mut sparse_bpp_mats = vec![SparseProbMat::<T>::new(); num_of_fasta_records];
  let mut max_bp_spans = vec![T::zero(); num_of_fasta_records];
  thread_pool.scoped(|scope| {
    for (bpp_mat, sparse_bpp_mat, max_bp_span, fasta_record) in multizip((
      bpp_mats.iter_mut(),
      sparse_bpp_mats.iter_mut(),
      max_bp_spans.iter_mut(),
      fasta_records.iter(),
    )) {
      let seq_len = fasta_record.seq.len();
      scope.execute(move || {
        *bpp_mat = mccaskill_algo(&fasta_record.seq[1..seq_len - 1], false);
        *sparse_bpp_mat = remove_small_bpps_from_bpp_mat::<T>(bpp_mat, min_bpp);
        *max_bp_span = get_max_bp_span::<T>(sparse_bpp_mat);
      });
    }
  });
  let mut prob_mats_with_rna_id_pairs = StaProbMatsWithRnaIdPairs::<T>::default();
  for rna_id_1 in 0..num_of_fasta_records {
    for rna_id_2 in rna_id_1 + 1..num_of_fasta_records {
      let rna_id_pair = (rna_id_1, rna_id_2);
      prob_mats_with_rna_id_pairs.insert(rna_id_pair, StaProbMats::<T>::origin());
    }
  }
  let feature_score_sets = FeatureCountSets::load_trained_score_params();
  thread_pool.scoped(|scope| {
    for (rna_id_pair, prob_mats) in prob_mats_with_rna_id_pairs.iter_mut() {
      let seq_pair = (
        &fasta_records[rna_id_pair.0].seq[..],
        &fasta_records[rna_id_pair.1].seq[..],
      );
      let seq_len_pair = (seq_pair.0.len(), seq_pair.1.len());
      let max_gap_num = offset_4_max_gap_num
        + T::from_usize(max(seq_len_pair.0, seq_len_pair.1) - min(seq_len_pair.0, seq_len_pair.1))
          .unwrap();
      let max_gap_num_4_il = max(
        min(max_gap_num, T::from_usize(MAX_GAP_NUM_4_IL).unwrap()),
        T::from_usize(MIN_GAP_NUM_4_IL).unwrap(),
      );
      let max_bp_span_pair = (max_bp_spans[rna_id_pair.0], max_bp_spans[rna_id_pair.1]);
      let bpp_mat_pair = (
        &sparse_bpp_mats[rna_id_pair.0],
        &sparse_bpp_mats[rna_id_pair.1],
      );
      let ref ref_2_feature_score_sets = feature_score_sets;
      scope.execute(move || {
        *prob_mats = io_algo_4_prob_mats::<T>(
          &seq_pair,
          ref_2_feature_score_sets,
          &max_bp_span_pair,
          max_gap_num,
          max_gap_num_4_il,
          &bpp_mat_pair,
          produces_access_probs,
          false,
          &mut FeatureCountSets::new(NEG_INFINITY),
        ).0;
      });
    }
  });
  let mut prob_mat_sets = vec![PctStaProbMats::<T>::origin(); num_of_fasta_records];
  thread_pool.scoped(|scope| {
    for (rna_id, prob_mats, bpp_mat) in multizip((
      0..num_of_fasta_records,
      prob_mat_sets.iter_mut(),
      bpp_mats.iter_mut(),
    )) {
      let ref ref_2_prob_mats_with_rna_id_pairs = prob_mats_with_rna_id_pairs;
      let seq_len = fasta_records[rna_id].seq.len();
      scope.execute(move || {
        *prob_mats = pct_of_prob_mats::<T>(
          ref_2_prob_mats_with_rna_id_pairs,
          rna_id,
          num_of_fasta_records,
          bpp_mat,
          seq_len,
          produces_access_probs,
        );
      });
    }
  });
  prob_mat_sets
}

pub fn rtrain<'a, T>(
  thread_pool: &mut Pool,
  train_data: &mut TrainData<T>,
  offset_4_max_gap_num: T,
  output_file_path: &Path,
)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send,
{
  let mut feature_score_sets = FeatureCountSets::new(0.);
  let mut old_feature_score_sets = feature_score_sets.clone();
  let mut old_log_loss = INFINITY;
  let mut log_losses = Probs::new();
  let mut count = 0;
  let mut regularizers = Regularizers::from_vec(vec![1.; feature_score_sets.get_len()]);
  loop {
    let ref ref_2_feature_score_sets = feature_score_sets;
    thread_pool.scoped(|scope| {
      for train_datum in train_data.iter_mut() {
        train_datum.expected_feature_count_sets = FeatureCountSets::new(NEG_INFINITY);
        let seq_pair = (&train_datum.seq_pair.0[..], &train_datum.seq_pair.1[..]);
        let seq_len_pair = (seq_pair.0.len(), seq_pair.1.len());
        let max_gap_num = offset_4_max_gap_num
          + T::from_usize(max(seq_len_pair.0, seq_len_pair.1) - min(seq_len_pair.0, seq_len_pair.1))
            .unwrap();
        let max_gap_num_4_il = max(
          min(max_gap_num, T::from_usize(MAX_GAP_NUM_4_IL_TRAIN).unwrap()),
          T::from_usize(MIN_GAP_NUM_4_IL_TRAIN).unwrap(),
        );
        let bpp_mat_pair = (&train_datum.bpp_mat_pair.0, &train_datum.bpp_mat_pair.1);
        let ref max_bp_span_pair = train_datum.max_bp_span_pair;
        let ref mut expected_feature_count_sets = train_datum.expected_feature_count_sets;
        let ref mut part_func = train_datum.part_func;
        scope.execute(move || {
          *part_func = io_algo_4_prob_mats::<T>(
            &seq_pair,
            ref_2_feature_score_sets,
            max_bp_span_pair,
            max_gap_num,
            max_gap_num_4_il,
            &bpp_mat_pair,
            false,
            true,
            expected_feature_count_sets,
          ).1;
        });
      }
    });
    feature_score_sets.update(&train_data, &mut regularizers);
    let log_loss = feature_score_sets.get_log_loss(&train_data[..], &regularizers);
    if log_loss > old_log_loss {
      feature_score_sets = old_feature_score_sets.clone();
      break;
    }
    log_losses.push(log_loss);
    old_feature_score_sets = feature_score_sets.clone();
    old_log_loss = log_loss;
    print!("\rEpoch {} finished", count + 1);
    stdout().flush().unwrap();
    count += 1;
  }
  write_feature_score_sets_trained(&feature_score_sets);
  write_log_losses(&log_losses, output_file_path);
}

pub fn remove_gaps(seq: &Seq) -> Seq {
  seq.iter().filter(|&&x| x != PSEUDO_BASE).map(|&x| x).collect()
}

pub fn convert_with_gaps<'a>(seq: &'a [u8]) -> Seq {
  let mut new_seq = Seq::new();
  for &c in seq {
    let new_base = convert_char(c);
    new_seq.push(new_base);
  }
  new_seq
}

pub fn convert_char(c: u8) -> Base {
  match c {
    SMALL_A | BIG_A => A,
    SMALL_C | BIG_C => C,
    SMALL_G | BIG_G => G,
    SMALL_U | BIG_U => U,
    _ => {PSEUDO_BASE},
  }
}

pub fn get_mismatch_pair(seq: SeqSlice, pos_pair: &(usize, usize), is_closing: bool) -> (usize, usize) {
  let mut mismatch_pair = (PSEUDO_BASE, PSEUDO_BASE);
  if is_closing {
    for i in pos_pair.0 + 1 .. pos_pair.1 {
      let align_char = seq[i];
      if align_char != PSEUDO_BASE {
        mismatch_pair.0 = align_char;
        break;
      }
    }
    for i in (pos_pair.0 + 1 .. pos_pair.1).rev() {
      let align_char = seq[i];
      if align_char != PSEUDO_BASE {
        mismatch_pair.1 = align_char;
        break;
      }
    }
  } else {
    for i in 0 .. pos_pair.0 {
      let align_char = seq[i];
      if align_char != PSEUDO_BASE {
        mismatch_pair.0 = align_char;
        break;
      }
    }
    let seq_len = seq.len();
    for i in pos_pair.1 + 1 .. seq_len {
      let align_char = seq[i];
      if align_char != PSEUDO_BASE {
        mismatch_pair.1 = align_char;
        break;
      }
    }
  }
  mismatch_pair
}

pub fn get_hairpin_loop_length(seq: SeqSlice, pos_pair: &(usize, usize)) -> usize {
  let mut hairpin_loop_length = 0;
  for i in pos_pair.0 + 1 .. pos_pair.1 {
    let align_char = seq[i];
    if align_char == PSEUDO_BASE {
      continue;
    }
    hairpin_loop_length += 1;
  }
  hairpin_loop_length
}

pub fn get_2loop_length_pair(seq: SeqSlice, pos_pair_closing_loop: &(usize, usize), pos_pair_in_loop: &(usize, usize)) -> (usize, usize) {
  let mut twoloop_length_pair = (0, 0);
  for i in pos_pair_closing_loop.0 + 1 .. pos_pair_in_loop.0 {
    let align_char = seq[i];
    if align_char == PSEUDO_BASE {
      continue;
    }
    twoloop_length_pair.0 += 1;
  }
  for i in pos_pair_in_loop.1 + 1 .. pos_pair_closing_loop.1 {
    let align_char = seq[i];
    if align_char == PSEUDO_BASE {
      continue;
    }
    twoloop_length_pair.1 += 1;
  }
  twoloop_length_pair
}

pub fn convert_vec_2_struct(feature_counts: &FeatureCounts, uses_cumulative_feature_counts: bool) -> FeatureCountSets {
  let mut f = FeatureCountSets::new(0.);
  let mut offset = 0;
  let len = f.hairpin_loop_length_counts.len();
  for i in 0 .. len {
    let count = feature_counts[offset + i];
    if uses_cumulative_feature_counts {
      f.hairpin_loop_length_counts_cumulative[i] = count;
    } else {
      f.hairpin_loop_length_counts[i] = count;
    }
  }
  offset += len;
  let len = f.bulge_loop_length_counts.len();
  for i in 0 .. len {
    let count = feature_counts[offset + i];
    if uses_cumulative_feature_counts {
      f.bulge_loop_length_counts_cumulative[i] = count;
    } else {
      f.bulge_loop_length_counts[i] = count;
    }
  }
  offset += len;
  let len = f.interior_loop_length_counts.len();
  for i in 0 .. len {
    let count = feature_counts[offset + i];
    if uses_cumulative_feature_counts {
      f.interior_loop_length_counts_cumulative[i] = count;
    } else {
      f.interior_loop_length_counts[i] = count;
    }
  }
  offset += len;
  let len = f.interior_loop_length_counts_symm.len();
  for i in 0 .. len {
    let count = feature_counts[offset + i];
    if uses_cumulative_feature_counts {
      f.interior_loop_length_counts_symm_cumulative[i] = count;
    } else {
      f.interior_loop_length_counts_symm[i] = count;
    }
  }
  offset += len;
  let len = f.interior_loop_length_counts_asymm.len();
  for i in 0 .. len {
    let count = feature_counts[offset + i];
    if uses_cumulative_feature_counts {
      f.interior_loop_length_counts_asymm_cumulative[i] = count;
    } else {
      f.interior_loop_length_counts_asymm[i] = count;
    }
  }
  offset += len;
  let len = f.stack_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      for k in 0 .. len {
        for l in 0 .. len {
          f.stack_count_mat[i][j][k][l] = feature_counts[offset + i * len.pow(3) + j * len.pow(2) + k * len + l];
        }
      }
    }
  }
  offset += len.pow(4);
  let len = f.terminal_mismatch_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      for k in 0 .. len {
        for l in 0 .. len {
          f.terminal_mismatch_count_mat[i][j][k][l] = feature_counts[offset + i * len.pow(3) + j * len.pow(2) + k * len + l];
        }
      }
    }
  }
  offset += len.pow(4);
  let len = f.left_dangle_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      for k in 0 .. len {
        f.left_dangle_count_mat[i][j][k] = feature_counts[offset + i * len.pow(2) + j * len + k];
      }
    }
  }
  offset += len.pow(3);
  let len = f.right_dangle_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      for k in 0 .. len {
        f.right_dangle_count_mat[i][j][k] = feature_counts[offset + i * len.pow(2) + j * len + k];
      }
    }
  }
  offset += len.pow(3);
  let len = f.helix_end_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      f.helix_end_count_mat[i][j] = feature_counts[offset + i * len + j];
    }
  }
  offset += len.pow(2);
  let len = f.base_pair_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      f.base_pair_count_mat[i][j] = feature_counts[offset + i * len + j];
    }
  }
  offset += len.pow(2);
  let len = f.interior_loop_length_count_mat_explicit.len();
  for i in 0 .. len {
    for j in 0 .. len {
      f.interior_loop_length_count_mat_explicit[i][j] = feature_counts[offset + i * len + j];
    }
  }
  offset += len.pow(2);
  let len = f.bulge_loop_0x1_length_counts.len();
  for i in 0 .. len {
    f.bulge_loop_0x1_length_counts[i] = feature_counts[offset + i];
  }
  offset += len;
  let len = f.interior_loop_1x1_length_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      f.interior_loop_1x1_length_count_mat[i][j] = feature_counts[offset + i * len + j];
    }
  }
  offset += len.pow(2);
  f.multi_loop_base_count = feature_counts[offset];
  offset += 1;
  f.multi_loop_basepairing_count = feature_counts[offset];
  offset += 1;
  f.multi_loop_accessible_baseunpairing_count = feature_counts[offset];
  offset += 1;
  f.external_loop_accessible_basepairing_count = feature_counts[offset];
  offset += 1;
  f.external_loop_accessible_baseunpairing_count = feature_counts[offset];
  offset += 1;
  let len = f.basepair_align_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      for k in 0 .. len {
        for l in 0 .. len {
          f.basepair_align_count_mat[i][j][k][l] = feature_counts[offset + i * len.pow(3) + j * len.pow(2) + k * len + l];
        }
      }
    }
  }
  offset += len.pow(4);
  let len = f.loop_align_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      f.loop_align_count_mat[i][j] = feature_counts[offset + i * len + j];
    }
  }
  offset += len.pow(2);
  f.opening_gap_count = feature_counts[offset];
  offset += 1;
  f.extending_gap_count = feature_counts[offset];
  f
}

pub fn convert_struct_2_vec(feature_count_sets: &FeatureCountSets, uses_cumulative_feature_counts: bool) -> FeatureCounts {
  let f = feature_count_sets;
  let mut feature_counts = vec![0.; f.get_len()];
  let mut offset = 0;
  let len = f.hairpin_loop_length_counts.len();
  for i in 0 .. len {
    feature_counts[offset + i] = if uses_cumulative_feature_counts {
      f.hairpin_loop_length_counts_cumulative[i]
    } else {
      f.hairpin_loop_length_counts[i]
    };
  }
  offset += len;
  let len = f.bulge_loop_length_counts.len();
  for i in 0 .. len {
    feature_counts[offset + i] = if uses_cumulative_feature_counts {
      f.bulge_loop_length_counts_cumulative[i]
    } else {
      f.bulge_loop_length_counts[i]
    };
  }
  offset += len;
  let len = f.interior_loop_length_counts.len();
  for i in 0 .. len {
    feature_counts[offset + i] = if uses_cumulative_feature_counts {
      f.interior_loop_length_counts_cumulative[i]
    } else {
      f.interior_loop_length_counts[i]
    };
  }
  offset += len;
  let len = f.interior_loop_length_counts_symm.len();
  for i in 0 .. len {
    feature_counts[offset + i] = if uses_cumulative_feature_counts {
      f.interior_loop_length_counts_symm_cumulative[i]
    } else {
      f.interior_loop_length_counts_symm[i]
    };
  }
  offset += len;
  let len = f.interior_loop_length_counts_asymm.len();
  for i in 0 .. len {
    feature_counts[offset + i] = if uses_cumulative_feature_counts {
      f.interior_loop_length_counts_asymm_cumulative[i]
    } else {
      f.interior_loop_length_counts_asymm[i]
    };
  }
  offset += len;
  let len = f.stack_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      for k in 0 .. len {
        for l in 0 .. len {
          feature_counts[offset + i * len.pow(3) + j * len.pow(2) + k * len + l] = f.stack_count_mat[i][j][k][l];
        }
      }
    }
  }
  offset += len.pow(4);
  let len = f.terminal_mismatch_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      for k in 0 .. len {
        for l in 0 .. len {
          feature_counts[offset + i * len.pow(3) + j * len.pow(2) + k * len + l] = f.terminal_mismatch_count_mat[i][j][k][l];
        }
      }
    }
  }
  offset += len.pow(4);
  let len = f.left_dangle_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      for k in 0 .. len {
        feature_counts[offset + i * len.pow(2) + j * len + k] = f.left_dangle_count_mat[i][j][k];
      }
    }
  }
  offset += len.pow(3);
  let len = f.right_dangle_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      for k in 0 .. len {
        feature_counts[offset + i * len.pow(2) + j * len + k] = f.right_dangle_count_mat[i][j][k];
      }
    }
  }
  offset += len.pow(3);
  let len = f.helix_end_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      feature_counts[offset + i * len + j] = f.helix_end_count_mat[i][j];
    }
  }
  offset += len.pow(2);
  let len = f.base_pair_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      feature_counts[offset + i * len + j] = f.base_pair_count_mat[i][j];
    }
  }
  offset += len.pow(2);
  let len = f.interior_loop_length_count_mat_explicit.len();
  for i in 0 .. len {
    for j in 0 .. len {
      feature_counts[offset + i * len + j] = f.interior_loop_length_count_mat_explicit[i][j];
    }
  }
  offset += len.pow(2);
  let len = f.bulge_loop_0x1_length_counts.len();
  for i in 0 .. len {
    feature_counts[offset + i] = f.bulge_loop_0x1_length_counts[i];
  }
  offset += len;
  let len = f.interior_loop_1x1_length_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      feature_counts[offset + i * len + j] = f.interior_loop_1x1_length_count_mat[i][j];
    }
  }
  offset += len.pow(2);
  feature_counts[offset] = f.multi_loop_base_count;
  offset += 1;
  feature_counts[offset] = f.multi_loop_basepairing_count;
  offset += 1;
  feature_counts[offset] = f.multi_loop_accessible_baseunpairing_count;
  offset += 1;
  feature_counts[offset] = f.external_loop_accessible_basepairing_count;
  offset += 1;
  feature_counts[offset] = f.external_loop_accessible_baseunpairing_count;
  offset += 1;
  let len = f.basepair_align_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      for k in 0 .. len {
        for l in 0 .. len {
          feature_counts[offset + i * len.pow(3) + j * len.pow(2) + k * len + l] = f.basepair_align_count_mat[i][j][k][l];
        }
      }
    }
  }
  offset += len.pow(4);
  let len = f.loop_align_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      feature_counts[offset + i * len + j] = f.loop_align_count_mat[i][j];
    }
  }
  offset += len.pow(2);
  feature_counts[offset] = f.opening_gap_count;
  offset += 1;
  feature_counts[offset] = f.extending_gap_count;
  Array::from(feature_counts)
}

pub fn convert_feature_counts_2_bfgs_feature_counts(feature_counts: &FeatureCounts) -> BfgsFeatureCounts {
  let vec = feature_counts.to_vec().iter().map(|x| *x as BfgsFeatureCount).collect();
  BfgsFeatureCounts::from_vec(vec)
}

pub fn convert_bfgs_feature_counts_2_feature_counts(feature_counts: &BfgsFeatureCounts) -> FeatureCounts {
  let vec = feature_counts.to_vec().iter().map(|x| *x as FeatureCount).collect();
  FeatureCounts::from_vec(vec)
}

pub fn get_regularizer(group_size: usize, squared_sum: FeatureCount) -> Regularizer {
  (group_size as FeatureCount / 2. + GAMMA_DIST_ALPHA) / (squared_sum / 2. + GAMMA_DIST_BETA)
}

pub fn get_regularizer_grad_comp(group_size: usize, squared_sum: FeatureCount, term: FeatureCount) -> Regularizer {
  let denom = squared_sum / 2. + GAMMA_DIST_BETA;
  - (group_size as FeatureCount / 2. + GAMMA_DIST_ALPHA) * 2. * term / (denom * denom)
}

pub fn write_feature_score_sets_trained(feature_score_sets: &FeatureCountSets) {
  let mut writer_2_trained_feature_score_sets_file = BufWriter::new(File::create(TRAINED_FEATURE_SCORE_SETS_FILE_PATH).unwrap());
  let mut buf_4_writer_2_trained_feature_score_sets_file = String::from("use FeatureCountSets;\nimpl FeatureCountSets {{\npub fn load_trained_score_params() -> FeatureCountSets {{\nFeatureCountSets {{\nhairpin_loop_length_counts: ");
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nbulge_loop_length_counts: ", &feature_score_sets.hairpin_loop_length_counts));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninterior_loop_length_counts: ", &feature_score_sets.bulge_loop_length_counts));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninterior_loop_length_counts_symm: ", &feature_score_sets.interior_loop_length_counts));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninterior_loop_length_counts_asymm: ", &feature_score_sets.interior_loop_length_counts_symm));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nstack_count_mat: ", &feature_score_sets.interior_loop_length_counts_asymm));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nterminal_mismatch_count_mat: ", &feature_score_sets.stack_count_mat));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nleft_dangle_count_mat: ", &feature_score_sets.terminal_mismatch_count_mat));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nright_dangle_count_mat: ", &feature_score_sets.left_dangle_count_mat));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nhelix_end_count_mat: ", &feature_score_sets.right_dangle_count_mat));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nbase_pair_count_mat: ", &feature_score_sets.helix_end_count_mat));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninterior_loop_length_count_mat_explicit: ", &feature_score_sets.base_pair_count_mat));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nbulge_loop_0x1_length_counts: ", &feature_score_sets.interior_loop_length_count_mat_explicit));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninterior_loop_1x1_length_count_mat: ", &feature_score_sets.bulge_loop_0x1_length_counts));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nmulti_loop_base_count: ", &feature_score_sets.interior_loop_1x1_length_count_mat));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nmulti_loop_basepairing_count: ", feature_score_sets.multi_loop_base_count));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nmulti_loop_accessible_baseunpairing_count: ", feature_score_sets.multi_loop_basepairing_count));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nexternal_loop_accessible_basepairing_count: ", feature_score_sets.multi_loop_accessible_baseunpairing_count));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nexternal_loop_accessible_baseunpairing_count: ", feature_score_sets.external_loop_accessible_basepairing_count));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nbasepair_align_count_mat: ", feature_score_sets.external_loop_accessible_baseunpairing_count));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nloop_align_count_mat: ", &feature_score_sets.basepair_align_count_mat));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nopening_gap_count: ", &feature_score_sets.loop_align_count_mat));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nextending_gap_count: ", feature_score_sets.opening_gap_count));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nhairpin_loop_length_counts_cumulative: ", feature_score_sets.extending_gap_count));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nbulge_loop_length_counts_cumulative: ", &feature_score_sets.hairpin_loop_length_counts_cumulative));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninterior_loop_length_counts_cumulative: ", &feature_score_sets.bulge_loop_length_counts_cumulative));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninterior_loop_length_counts_symm_cumulative: ", &feature_score_sets.interior_loop_length_counts_cumulative));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninterior_loop_length_counts_asymm_cumulative: ", &feature_score_sets.interior_loop_length_counts_symm_cumulative));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},", &feature_score_sets.interior_loop_length_counts_asymm_cumulative));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&String::from("\n}\n}\n}"));
  let _ = writer_2_trained_feature_score_sets_file.write_all(buf_4_writer_2_trained_feature_score_sets_file.as_bytes());
}

pub fn write_log_losses(log_losses: &Probs, output_file_path: &Path) {
  let mut writer_2_output_file = BufWriter::new(File::create(output_file_path).unwrap());
  let mut buf_4_writer_2_output_file = String::new();
  for log_loss in log_losses {
    buf_4_writer_2_output_file.push_str(&format!("{}\n", log_loss));
  }
  let _ = writer_2_output_file.write_all(buf_4_writer_2_output_file.as_bytes());
}

pub fn write_prob_mat_sets<T>(
  output_dir_path: &Path,
  prob_mat_sets: &ProbMatSets<T>,
  produces_access_probs: bool,
) where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Display + Ord,
{
  if !output_dir_path.exists() {
    let _ = create_dir(output_dir_path);
  }
  let bpp_mat_file_path = output_dir_path.join(BPP_MAT_FILE_NAME);
  let mut writer_2_bpp_mat_file = BufWriter::new(File::create(bpp_mat_file_path).unwrap());
  let mut buf_4_writer_2_bpp_mat_file = String::new();
  for (rna_id, prob_mats) in prob_mat_sets.iter().enumerate() {
    let mut buf_4_rna_id = format!("\n\n>{}\n", rna_id);
    for (&(i, j), &bpp) in prob_mats.bpp_mat.iter() {
      buf_4_rna_id.push_str(&format!("{},{},{} ", i - T::one(), j - T::one(), bpp));
    }
    buf_4_writer_2_bpp_mat_file.push_str(&buf_4_rna_id);
  }
  let _ = writer_2_bpp_mat_file.write_all(buf_4_writer_2_bpp_mat_file.as_bytes());
  let upp_mat_file_path = output_dir_path.join(UPP_MAT_FILE_NAME);
  let mut writer_2_upp_mat_file = BufWriter::new(File::create(upp_mat_file_path).unwrap());
  let mut buf_4_writer_2_upp_mat_file = String::new();
  for (rna_id, prob_mats) in prob_mat_sets.iter().enumerate() {
    let seq_len = prob_mats.upp_mat.len();
    let mut buf_4_rna_id = format!("\n\n>{}\n", rna_id);
    for (i, &upp) in prob_mats.upp_mat.iter().enumerate() {
      if i == 0 || i == seq_len - 1 {
        continue;
      }
      buf_4_rna_id.push_str(&format!("{},{} ", i - 1, upp));
    }
    buf_4_writer_2_upp_mat_file.push_str(&buf_4_rna_id);
  }
  let _ = writer_2_upp_mat_file.write_all(buf_4_writer_2_upp_mat_file.as_bytes());
  let bpp_mat_file_path = output_dir_path.join(BPP_MAT_ON_SS_FILE_NAME);
  let mut writer_2_bpp_mat_file = BufWriter::new(File::create(bpp_mat_file_path).unwrap());
  let mut buf_4_writer_2_bpp_mat_file = String::new();
  for (rna_id, prob_mats) in prob_mat_sets.iter().enumerate() {
    let mut buf_4_rna_id = format!("\n\n>{}\n", rna_id);
    for (&(i, j), &bpp) in prob_mats.bpp_mat_on_ss.iter() {
      buf_4_rna_id.push_str(&format!("{},{},{} ", i, j, bpp));
    }
    buf_4_writer_2_bpp_mat_file.push_str(&buf_4_rna_id);
  }
  let _ = writer_2_bpp_mat_file.write_all(buf_4_writer_2_bpp_mat_file.as_bytes());
  if produces_access_probs {
    let bpp_mat_file_path = output_dir_path.join(MAX_BPP_MAT_FILE_NAME);
    let mut writer_2_bpp_mat_file = BufWriter::new(File::create(bpp_mat_file_path).unwrap());
    let mut buf_4_writer_2_bpp_mat_file = String::new();
    for (rna_id, prob_mats) in prob_mat_sets.iter().enumerate() {
      let mut buf_4_rna_id = format!("\n\n>{}\n", rna_id);
      for (&(i, j), &bpp) in prob_mats.max_bpp_mat.iter() {
        buf_4_rna_id.push_str(&format!("{},{},{} ", i - T::one(), j - T::one(), bpp));
      }
      buf_4_writer_2_bpp_mat_file.push_str(&buf_4_rna_id);
    }
    let _ = writer_2_bpp_mat_file.write_all(buf_4_writer_2_bpp_mat_file.as_bytes());
    let bpp_mat_file_path = output_dir_path.join(ACCESS_BPP_MAT_ON_2L_FILE_NAME);
    let mut writer_2_bpp_mat_file = BufWriter::new(File::create(bpp_mat_file_path).unwrap());
    let mut buf_4_writer_2_bpp_mat_file = String::new();
    for (rna_id, prob_mats) in prob_mat_sets.iter().enumerate() {
      let mut buf_4_rna_id = format!("\n\n>{}\n", rna_id);
      for (&(i, j), &bpp) in prob_mats.access_bpp_mat_4_2l.iter() {
        buf_4_rna_id.push_str(&format!("{},{},{} ", i - T::one(), j - T::one(), bpp));
      }
      buf_4_writer_2_bpp_mat_file.push_str(&buf_4_rna_id);
    }
    let _ = writer_2_bpp_mat_file.write_all(buf_4_writer_2_bpp_mat_file.as_bytes());
    let bpp_mat_file_path = output_dir_path.join(ACCESS_BPP_MAT_ON_ML_FILE_NAME);
    let mut writer_2_bpp_mat_file = BufWriter::new(File::create(bpp_mat_file_path).unwrap());
    let mut buf_4_writer_2_bpp_mat_file = String::new();
    for (rna_id, prob_mats) in prob_mat_sets.iter().enumerate() {
      let mut buf_4_rna_id = format!("\n\n>{}\n", rna_id);
      for (&(i, j), &bpp) in prob_mats.access_bpp_mat_4_ml.iter() {
        buf_4_rna_id.push_str(&format!("{},{},{} ", i - T::one(), j - T::one(), bpp));
      }
      buf_4_writer_2_bpp_mat_file.push_str(&buf_4_rna_id);
    }
    let _ = writer_2_bpp_mat_file.write_all(buf_4_writer_2_bpp_mat_file.as_bytes());
    let bpp_mat_file_path = output_dir_path.join(BPP_MAT_ON_EL_FILE_NAME);
    let mut writer_2_bpp_mat_file = BufWriter::new(File::create(bpp_mat_file_path).unwrap());
    let mut buf_4_writer_2_bpp_mat_file = String::new();
    for (rna_id, prob_mats) in prob_mat_sets.iter().enumerate() {
      let mut buf_4_rna_id = format!("\n\n>{}\n", rna_id);
      for (&(i, j), &bpp) in prob_mats.bpp_mat_4_el.iter() {
        buf_4_rna_id.push_str(&format!("{},{},{} ", i - T::one(), j - T::one(), bpp));
      }
      buf_4_writer_2_bpp_mat_file.push_str(&buf_4_rna_id);
    }
    let _ = writer_2_bpp_mat_file.write_all(buf_4_writer_2_bpp_mat_file.as_bytes());
    let upp_mat_file_path = output_dir_path.join(MAX_UPP_MAT_FILE_NAME);
    let mut writer_2_upp_mat_file = BufWriter::new(File::create(upp_mat_file_path).unwrap());
    let mut buf_4_writer_2_upp_mat_file = String::new();
    for (rna_id, prob_mats) in prob_mat_sets.iter().enumerate() {
      let seq_len = prob_mats.max_upp_mat.len();
      let mut buf_4_rna_id = format!("\n\n>{}\n", rna_id);
      for (i, &upp) in prob_mats.max_upp_mat.iter().enumerate() {
        if i == 0 || i == seq_len - 1 {
          continue;
        }
        buf_4_rna_id.push_str(&format!("{},{} ", i - 1, upp));
      }
      buf_4_writer_2_upp_mat_file.push_str(&buf_4_rna_id);
    }
    let _ = writer_2_upp_mat_file.write_all(buf_4_writer_2_upp_mat_file.as_bytes());
    let upp_mat_file_path = output_dir_path.join(UPP_MAT_ON_HL_FILE_NAME);
    let mut writer_2_upp_mat_file = BufWriter::new(File::create(upp_mat_file_path).unwrap());
    let mut buf_4_writer_2_upp_mat_file = String::new();
    for (rna_id, prob_mats) in prob_mat_sets.iter().enumerate() {
      let seq_len = prob_mats.upp_mat_4_hl.len();
      let mut buf_4_rna_id = format!("\n\n>{}\n", rna_id);
      for (i, &upp) in prob_mats.upp_mat_4_hl.iter().enumerate() {
        if i == 0 || i == seq_len - 1 {
          continue;
        }
        buf_4_rna_id.push_str(&format!("{},{} ", i - 1, upp));
      }
      buf_4_writer_2_upp_mat_file.push_str(&buf_4_rna_id);
    }
    let _ = writer_2_upp_mat_file.write_all(buf_4_writer_2_upp_mat_file.as_bytes());
    let upp_mat_file_path = output_dir_path.join(UPP_MAT_ON_2L_FILE_NAME);
    let mut writer_2_upp_mat_file = BufWriter::new(File::create(upp_mat_file_path).unwrap());
    let mut buf_4_writer_2_upp_mat_file = String::new();
    for (rna_id, prob_mats) in prob_mat_sets.iter().enumerate() {
      let seq_len = prob_mats.upp_mat_4_2l.len();
      let mut buf_4_rna_id = format!("\n\n>{}\n", rna_id);
      for (i, &upp) in prob_mats.upp_mat_4_2l.iter().enumerate() {
        if i == 0 || i == seq_len - 1 {
          continue;
        }
        buf_4_rna_id.push_str(&format!("{},{} ", i - 1, upp));
      }
      buf_4_writer_2_upp_mat_file.push_str(&buf_4_rna_id);
    }
    let _ = writer_2_upp_mat_file.write_all(buf_4_writer_2_upp_mat_file.as_bytes());
    let upp_mat_file_path = output_dir_path.join(UPP_MAT_ON_ML_FILE_NAME);
    let mut writer_2_upp_mat_file = BufWriter::new(File::create(upp_mat_file_path).unwrap());
    let mut buf_4_writer_2_upp_mat_file = String::new();
    for (rna_id, prob_mats) in prob_mat_sets.iter().enumerate() {
      let seq_len = prob_mats.upp_mat_4_ml.len();
      let mut buf_4_rna_id = format!("\n\n>{}\n", rna_id);
      for (i, &upp) in prob_mats.upp_mat_4_ml.iter().enumerate() {
        if i == 0 || i == seq_len - 1 {
          continue;
        }
        buf_4_rna_id.push_str(&format!("{},{} ", i - 1, upp));
      }
      buf_4_writer_2_upp_mat_file.push_str(&buf_4_rna_id);
    }
    let _ = writer_2_upp_mat_file.write_all(buf_4_writer_2_upp_mat_file.as_bytes());
    let upp_mat_file_path = output_dir_path.join(UPP_MAT_ON_EL_FILE_NAME);
    let mut writer_2_upp_mat_file = BufWriter::new(File::create(upp_mat_file_path).unwrap());
    let mut buf_4_writer_2_upp_mat_file = String::new();
    for (rna_id, prob_mats) in prob_mat_sets.iter().enumerate() {
      let seq_len = prob_mats.upp_mat_4_el.len();
      let mut buf_4_rna_id = format!("\n\n>{}\n", rna_id);
      for (i, &upp) in prob_mats.upp_mat_4_el.iter().enumerate() {
        if i == 0 || i == seq_len - 1 {
          continue;
        }
        buf_4_rna_id.push_str(&format!("{},{} ", i - 1, upp));
      }
      buf_4_writer_2_upp_mat_file.push_str(&buf_4_rna_id);
    }
    let _ = writer_2_upp_mat_file.write_all(buf_4_writer_2_upp_mat_file.as_bytes());
  }
}
