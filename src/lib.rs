extern crate rna_algos;
extern crate bio;
extern crate getopts;
extern crate hashbrown;
extern crate itertools;
extern crate num_cpus;
extern crate scoped_threadpool;
extern crate bfgs;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
#[macro_use]
extern crate lazy_static;

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
pub use std::f32::INFINITY;
pub use std::io::stdout;
pub use ndarray::Array;
pub use ndarray_rand::RandomExt;
pub use ndarray_rand::rand_distr::{Normal, Distribution};
pub use rand::thread_rng;

pub type RealBpScoreParamSetPair<T> = (BpScoreParamSets<T>, BpScoreParamSets<T>);
pub type BpScoreParamSetPair<'a, T> = (&'a BpScoreParamSets<T>, &'a BpScoreParamSets<T>);
pub type BpScores<T> = HashMap<PosPair<T>, FeatureCount>;
pub type TwoloopScores<T> = HashMap<PosQuadruple<T>, FeatureCount>;
pub type PosQuadrupleMat<T> = HashSet<PosQuadruple<T>>;
pub type PosPairMatSet<T> = HashMap<PosPair<T>, PosPairMat<T>>;
pub type PosPairMat<T> = HashSet<PosPair<T>>;
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
pub type AlignCountMat = LoopAlignCountMat;
pub type InsertCounts = [FeatureCount; NUM_OF_BASES];
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
pub struct BpScoreParamSets<T> {
  pub hairpin_loop_scores: BpScores<T>,
  pub twoloop_scores: TwoloopScores<T>,
  pub multi_loop_closing_bp_scores: BpScores<T>,
  pub multi_loop_accessible_bp_scores: BpScores<T>,
  pub external_loop_accessible_bp_scores: BpScores<T>,
}
#[derive(Clone)]
pub struct TrainDatum<T> {
  pub seq_pair: RealSeqPair,
  pub observed_feature_count_sets: FeatureCountSets,
  pub expected_feature_count_sets: FeatureCountSets,
  pub bpp_mat_pair: SparseProbMatPair<T>,
  pub max_bp_span_pair: (T, T),
  pub part_func: Prob,
  pub forward_pos_pair_mat_set: PosPairMatSet<T>,
  pub backward_pos_pair_mat_set: PosPairMatSet<T>,
  pub pos_quadruple_mat: PosQuadrupleMat<T>,
  pub bp_score_param_set_pair: RealBpScoreParamSetPair<T>,
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
  pub match_2_match_count: FeatureCount,
  pub match_2_insert_count: FeatureCount,
  pub insert_extend_count: FeatureCount,
  pub insert_switch_count: FeatureCount,
  pub insert_counts: InsertCounts,
  pub align_count_mat: AlignCountMat,
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
}
pub type TmpPartFuncSetMat4El<T> = HashMap<PosPair<T>, TmpPartFuncs>;
#[derive(Clone)]
pub struct StaPartFuncMats<T> {
  pub part_func_4d_mat_4_bpas: PartFunc4dMat<T>,
  pub part_func_4d_mat_4_bpas_accessible_on_els: PartFunc4dMat<T>,
  pub part_func_4d_mat_4_bpas_accessible_on_mls: PartFunc4dMat<T>,
  pub forward_part_func_set_mat_4_external_loop: TmpPartFuncSetMat4El<T>,
  pub backward_part_func_set_mat_4_external_loop: TmpPartFuncSetMat4El<T>,
  pub forward_tmp_part_func_set_mats_with_pos_pairs: TmpPartFuncSetMatsWithPosPairs<T>,
  pub backward_tmp_part_func_set_mats_with_pos_pairs: TmpPartFuncSetMatsWithPosPairs<T>,
  pub forward_2loop_tmp_part_func_set_mats_with_pos_quadruples: TmpPartFuncSetMatsWithPosQuadruples<T>,
  pub backward_2loop_tmp_part_func_set_mats_with_pos_quadruples: TmpPartFuncSetMatsWithPosQuadruples<T>,
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
pub type ProbMatPair<'a, T> = (&'a SparseProbMat<T>, &'a SparseProbMat<T>);
pub type NumOfThreads = u32;
pub struct StaProbMats<T> {
  pub bpp_mat_pair: SparseProbMatPair<T>,
  pub upp_mat_pair_4_hl: ProbSetPair,
  pub upp_mat_pair_4_2l: ProbSetPair,
  pub upp_mat_pair_4_ml: ProbSetPair,
  pub upp_mat_pair_4_el: ProbSetPair,
  pub bpp_mat_pair_2: ProbSetPair,
  pub basepair_align_prob_mat: Prob4dMat<T>,
  pub loop_align_prob_mat: SparseProbMat<T>,
}
#[derive(Clone)]
pub struct PctStaProbMats<T> {
  pub bpp_mat: SparseProbMat<T>,
  pub bpp_mat_on_ss: SparseProbMat<T>,
  pub upp_mat_4_hl: Probs,
  pub upp_mat_4_2l: Probs,
  pub upp_mat_4_ml: Probs,
  pub upp_mat_4_el: Probs,
  pub bpp_mat_2: Probs,
}
pub type RealProbSeqPair = (Probs, Probs);
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
#[derive(Clone)]
pub struct AlignProbMats<T> {
  pub loop_align_prob_mat: SparseProbMat<T>,
  pub basepair_align_prob_mat: Prob4dMat<T>,
  pub align_prob_mat: SparseProbMat<T>,
}
pub type AlignProbMatSetsWithRnaIdPairs<T> = HashMap<RnaIdPair, AlignProbMats<T>>;

impl<T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord> BpScoreParamSets<T> {
  pub fn new() -> BpScoreParamSets<T> {
    BpScoreParamSets {
      hairpin_loop_scores: BpScores::<T>::default(),
      twoloop_scores: TwoloopScores::<T>::default(),
      multi_loop_closing_bp_scores: BpScores::<T>::default(),
      multi_loop_accessible_bp_scores: BpScores::<T>::default(),
      external_loop_accessible_bp_scores: BpScores::<T>::default(),
    }
  }

  pub fn set_curr_params(feature_score_sets: &FeatureCountSets, seq: SeqSlice, bpp_mat: &SparseProbMat<T>) -> BpScoreParamSets<T> {
    let seq_len = seq.len();
    let mut bp_score_param_sets = BpScoreParamSets::<T>::new();
    for pos_pair in bpp_mat.keys() {
      let long_pos_pair = (pos_pair.0.to_usize().unwrap(), pos_pair.1.to_usize().unwrap());
      let hairpin_loop_score =
        get_consprob_hairpin_loop_score(feature_score_sets, seq, &long_pos_pair);
      if hairpin_loop_score > NEG_INFINITY {
        bp_score_param_sets.hairpin_loop_scores.insert(*pos_pair, hairpin_loop_score);
      }
      let multi_loop_closing_basepairing_score = get_consprob_multi_loop_closing_basepairing_score(
        feature_score_sets,
        seq,
        &long_pos_pair,
      );
      bp_score_param_sets.multi_loop_closing_bp_scores.insert(*pos_pair, multi_loop_closing_basepairing_score);
      let base_pair = (seq[long_pos_pair.1], seq[long_pos_pair.0]);
      let mismatch_pair = (seq[long_pos_pair.1 + 1], seq[long_pos_pair.0 - 1]);
      let left_dangle_score = if long_pos_pair.1 < seq_len - 2 {
        feature_score_sets.left_dangle_count_mat[base_pair.0][base_pair.1]
        [mismatch_pair.0]
      } else {
        0.
      };
      let right_dangle_score = if long_pos_pair.0 > 1 {
        feature_score_sets.right_dangle_count_mat[base_pair.0][base_pair.1]
        [mismatch_pair.1]
      } else {
        0.
      };
      let accessible_bp_shared_subscore = feature_score_sets.base_pair_count_mat[base_pair.0][base_pair.1];
      let helix_end_score = feature_score_sets.helix_end_count_mat[base_pair.0][base_pair.1];
      let accessible_bp_shared_score = accessible_bp_shared_subscore
        + helix_end_score;
      let accessible_bp_shared_score_with_dangle = accessible_bp_shared_score + left_dangle_score + right_dangle_score;
      let multi_loop_accessible_basepairing_score =
        get_consprob_multi_loop_accessible_basepairing_score(
          feature_score_sets,
          seq,
          &long_pos_pair,
          accessible_bp_shared_score_with_dangle,
        );
      bp_score_param_sets.multi_loop_accessible_bp_scores.insert(*pos_pair, multi_loop_accessible_basepairing_score);
      let external_loop_accessible_basepairing_score =
        get_consprob_external_loop_accessible_basepairing_score(
          feature_score_sets,
          accessible_bp_shared_score_with_dangle,
        );
      bp_score_param_sets.external_loop_accessible_bp_scores.insert(*pos_pair, external_loop_accessible_basepairing_score);
      let terminal_mismatch_score = if long_pos_pair.1 < seq_len - 2 && long_pos_pair.0 > 1 {feature_score_sets.terminal_mismatch_count_mat[base_pair.0][base_pair.1][mismatch_pair.0][mismatch_pair.1]} else {0.};
      for pos_pair_2 in bpp_mat.keys() {
        if !(pos_pair_2.0 < pos_pair.0 && pos_pair.1 < pos_pair_2.1) {continue;}
        let long_pos_pair_2 = (pos_pair_2.0.to_usize().unwrap(), pos_pair_2.1.to_usize().unwrap());
        if long_pos_pair.0 - long_pos_pair_2.0 - 1 + long_pos_pair_2.1 - long_pos_pair.1 - 1 > CONSPROB_MAX_TWOLOOP_LEN {
          continue;
        }
        let twoloop_score = get_consprob_twoloop_score(
          feature_score_sets,
          seq,
          &long_pos_pair_2,
          &long_pos_pair,
          accessible_bp_shared_subscore,
          helix_end_score,
          terminal_mismatch_score,
        );
        bp_score_param_sets.twoloop_scores.insert((pos_pair_2.0, pos_pair_2.1, pos_pair.0, pos_pair.1), twoloop_score);
      }
    }
    bp_score_param_sets
  }
}

impl FeatureCountSets {
  pub fn new(init_val: FeatureCount) -> FeatureCountSets {
    let init_vals = [init_val; NUM_OF_BASES];
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
      match_2_match_count: init_val,
      match_2_insert_count: init_val,
      insert_extend_count: init_val,
      insert_switch_count: init_val,
      insert_counts: init_vals,
      align_count_mat: twod_mat,
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
      + 1
      + 1
      + self.insert_counts.len()
      + self.align_count_mat.len().pow(2)
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
    let effective_group_size = NUM_OF_BASEPAIRINGS * NUM_OF_BASEPAIRINGS;
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
    let regularizer = get_regularizer(effective_group_size, squared_sum);
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
    let effective_group_size = NUM_OF_BASEPAIRINGS * len * len;
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
    let regularizer = get_regularizer(effective_group_size, squared_sum);
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
    let effective_group_size = NUM_OF_BASEPAIRINGS * len;
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
    let regularizer = get_regularizer(effective_group_size, squared_sum);
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
    let effective_group_size = NUM_OF_BASEPAIRINGS * len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          let count = self.right_dangle_count_mat[i][j][k];
          squared_sum += count * count;
        }
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
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
    let effective_group_size = NUM_OF_BASEPAIRINGS;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        let count = self.helix_end_count_mat[i][j];
        squared_sum += count * count;
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        regularizers_tmp[offset + i * len + j] = regularizer;
      }
    }
    offset += group_size;
    let len = self.base_pair_count_mat.len();
    let group_size = len.pow(2);
    let effective_group_size = NUM_OF_BASEPAIRINGS;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        let count = self.base_pair_count_mat[i][j];
        squared_sum += count * count;
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
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
    let effective_group_size = NUM_OF_BASEPAIRINGS * NUM_OF_BASEPAIRINGS;
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
    let regularizer = get_regularizer(effective_group_size, squared_sum);
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
    let regularizer = get_regularizer(1, self.match_2_match_count * self.match_2_match_count);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let regularizer = get_regularizer(1, self.match_2_insert_count * self.match_2_insert_count);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let regularizer = get_regularizer(1, self.insert_extend_count * self.insert_extend_count);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let regularizer = get_regularizer(1, self.insert_switch_count * self.insert_switch_count);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let len = self.insert_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0 .. len {
      let count = self.insert_counts[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.align_count_mat.len();
    let group_size = len.pow(2);
    let mut squared_sum = 0.;
    for i in 0 .. len {
      for j in 0 .. len {
        let count = self.align_count_mat[i][j];
        squared_sum += count * count;
      }
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0 .. len {
      for j in 0 .. len {
        regularizers_tmp[offset + i * len + j] = regularizer;
      }
    }
    *regularizers = Array1::from(regularizers_tmp);
  }

  pub fn update<T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord>(&mut self, train_data: &[TrainDatum<T>], regularizers: &mut Regularizers)
  {
    let f = |_: &BfgsFeatureCounts| {
      self.get_cost(&train_data[..], regularizers) as BfgsFeatureCount
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
      let obs_count = obs.match_2_match_count;
      let expect_count = expect.match_2_match_count;
      grad.match_2_match_count -= obs_count - expect_count;
      let obs_count = obs.match_2_insert_count;
      let expect_count = expect.match_2_insert_count;
      grad.match_2_insert_count -= obs_count - expect_count;
      let obs_count = obs.insert_extend_count;
      let expect_count = expect.insert_extend_count;
      grad.insert_extend_count -= obs_count - expect_count;
      let obs_count = obs.insert_switch_count;
      let expect_count = expect.insert_switch_count;
      grad.insert_switch_count -= obs_count - expect_count;
      for i in 0 .. NUM_OF_BASES {
        let obs_count = obs.insert_counts[i];
        let expect_count = expect.insert_counts[i];
        grad.insert_counts[i] -= obs_count - expect_count;
      }
      for i in 0 .. NUM_OF_BASES {
        for j in 0 .. NUM_OF_BASES {
          let dict_min_align = get_dict_min_align(&(i, j));
          let obs_count = obs.align_count_mat[dict_min_align.0][dict_min_align.1];
          let expect_count = expect.align_count_mat[dict_min_align.0][dict_min_align.1];
          grad.align_count_mat[i][j] -= obs_count - expect_count;
        }
      }
    }
    convert_struct_2_vec(&grad, false) + regularizers.clone() * feature_scores
  }

  pub fn get_cost<T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord>(&self, train_data: &[TrainDatum<T>], regularizers: &Regularizers) -> FeatureCount {
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

  pub fn rand_init(&mut self) {
    let len = self.get_len();
    let std_deviation = 1. / (len as FeatureCount).sqrt();
    let normal = Normal::new(0., std_deviation).unwrap();
    let mut thread_rng = thread_rng();
    for v in self.hairpin_loop_length_counts.iter_mut() {
      *v = normal.sample(&mut thread_rng);
    }
    for v in self.bulge_loop_length_counts.iter_mut() {
      *v = normal.sample(&mut thread_rng);
    }
    for v in self.interior_loop_length_counts.iter_mut() {
      *v = normal.sample(&mut thread_rng);
    }
    for v in self.interior_loop_length_counts_symm.iter_mut() {
      *v = normal.sample(&mut thread_rng);
    }
    for v in self.interior_loop_length_counts_asymm.iter_mut() {
      *v = normal.sample(&mut thread_rng);
    }
    let len = self.stack_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            if !is_canonical(&(k, l)) {continue;}
            let v = normal.sample(&mut thread_rng);
            if self.stack_count_mat[i][j][k][l] == 0. {
              self.stack_count_mat[i][j][k][l] = v;
            }
            if self.stack_count_mat[l][k][j][i] == 0. {
              self.stack_count_mat[l][k][j][i] = v;
            }
          }
        }
      }
    }
    let len = self.terminal_mismatch_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            let v = normal.sample(&mut thread_rng);
            self.terminal_mismatch_count_mat[i][j][k][l] = v;
          }
        }
      }
    }
    let len = self.left_dangle_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          let v = normal.sample(&mut thread_rng);
          self.left_dangle_count_mat[i][j][k] = v;
        }
      }
    }
    let len = self.right_dangle_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          let v = normal.sample(&mut thread_rng);
          self.right_dangle_count_mat[i][j][k] = v;
        }
      }
    }
    let len = self.helix_end_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        let v = normal.sample(&mut thread_rng);
        self.helix_end_count_mat[i][j] = v;
      }
    }
    let len = self.base_pair_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        let v = normal.sample(&mut thread_rng);
        if self.base_pair_count_mat[i][j] == 0. {
          self.base_pair_count_mat[i][j] = v;
        }
        if self.base_pair_count_mat[j][i] == 0. {
          self.base_pair_count_mat[j][i] = v;
        }
      }
    }
    let len = self.interior_loop_length_count_mat_explicit.len();
    for i in 0 .. len {
      for j in 0 .. len {
        let v = normal.sample(&mut thread_rng);
        if self.interior_loop_length_count_mat_explicit[i][j] == 0. {
          self.interior_loop_length_count_mat_explicit[i][j] = v;
        }
        if self.interior_loop_length_count_mat_explicit[j][i] == 0. {
          self.interior_loop_length_count_mat_explicit[j][i] = v;
        }
      }
    }
    for v in &mut self.bulge_loop_0x1_length_counts {
      *v = normal.sample(&mut thread_rng);
    }
    let len = self.interior_loop_1x1_length_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        let v = normal.sample(&mut thread_rng);
        if self.interior_loop_1x1_length_count_mat[i][j] == 0. {
          self.interior_loop_1x1_length_count_mat[i][j] = v;
        }
        if self.interior_loop_1x1_length_count_mat[j][i] == 0. {
          self.interior_loop_1x1_length_count_mat[j][i] = v;
        }
      }
    }
    self.multi_loop_base_count = normal.sample(&mut thread_rng);
    self.multi_loop_basepairing_count = normal.sample(&mut thread_rng);
    self.multi_loop_accessible_baseunpairing_count = normal.sample(&mut thread_rng);
    self.external_loop_accessible_basepairing_count = normal.sample(&mut thread_rng);
    self.external_loop_accessible_baseunpairing_count = normal.sample(&mut thread_rng);
    let len = self.basepair_align_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            if !is_canonical(&(k, l)) {continue;}
            let v = normal.sample(&mut thread_rng);
            if self.basepair_align_count_mat[i][j][k][l] == 0. {
              self.basepair_align_count_mat[i][j][k][l] = v;
            }
            if self.basepair_align_count_mat[k][l][i][j] == 0. {
              self.basepair_align_count_mat[k][l][i][j] = v;
            }
          }
        }
      }
    }
    let len = self.loop_align_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        let v = normal.sample(&mut thread_rng);
        if self.loop_align_count_mat[i][j] == 0. {
          self.loop_align_count_mat[i][j] = v;
        }
        if self.loop_align_count_mat[j][i] == 0. {
          self.loop_align_count_mat[j][i] = v;
        }
      }
    }
    self.match_2_match_count = normal.sample(&mut thread_rng);
    self.match_2_insert_count = normal.sample(&mut thread_rng);
    self.insert_extend_count = normal.sample(&mut thread_rng);
    self.insert_switch_count = normal.sample(&mut thread_rng);
    let len = self.insert_counts.len();
    for i in 0 .. len {
      let v = normal.sample(&mut thread_rng);
      if self.insert_counts[i] == 0. {
        self.insert_counts[i] = v;
      }
    }
    let len = self.align_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        let v = normal.sample(&mut thread_rng);
        if self.align_count_mat[i][j] == 0. {
          self.align_count_mat[i][j] = v;
        }
        if self.align_count_mat[j][i] == 0. {
          self.align_count_mat[j][i] = v;
        }
      }
    }
    self.accumulate();
  }

  pub fn transfer(&mut self) {
    for (v, &w) in self.hairpin_loop_length_counts.iter_mut().zip(CONTRA_HL_LENGTH_FES_AT_LEAST.iter()) {
      *v = w;
    }
    for (v, &w) in self.bulge_loop_length_counts.iter_mut().zip(CONTRA_BL_LENGTH_FES_AT_LEAST.iter()) {
      *v = w;
    }
    for (v, &w) in self.interior_loop_length_counts.iter_mut().zip(CONTRA_IL_LENGTH_FES_AT_LEAST.iter()) {
      *v = w;
    }
    for (v, &w) in self.interior_loop_length_counts_symm.iter_mut().zip(CONTRA_IL_SYMM_LENGTH_FES_AT_LEAST.iter()) {
      *v = w;
    }
    for (v, &w) in self.interior_loop_length_counts_asymm.iter_mut().zip(CONTRA_IL_ASYMM_LENGTH_FES_AT_LEAST.iter()) {
      *v = w;
    }
    let len = self.stack_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            if !is_canonical(&(k, l)) {continue;}
            self.stack_count_mat[i][j][k][l] = CONTRA_STACK_FES[i][j][k][l];
          }
        }
      }
    }
    let len = self.terminal_mismatch_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            self.terminal_mismatch_count_mat[i][j][k][l] = CONTRA_TERMINAL_MISMATCH_FES[i][j][k][l];
          }
        }
      }
    }
    let len = self.left_dangle_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          self.left_dangle_count_mat[i][j][k] = CONTRA_LEFT_DANGLE_FES[i][j][k];
        }
      }
    }
    let len = self.right_dangle_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          self.right_dangle_count_mat[i][j][k] = CONTRA_RIGHT_DANGLE_FES[i][j][k];
        }
      }
    }
    let len = self.helix_end_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        self.helix_end_count_mat[i][j] = CONTRA_HELIX_CLOSING_FES[i][j];
      }
    }
    let len = self.base_pair_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        self.base_pair_count_mat[i][j] = CONTRA_BASE_PAIR_FES[i][j];
      }
    }
    let len = self.interior_loop_length_count_mat_explicit.len();
    for i in 0 .. len {
      for j in 0 .. len {
        self.interior_loop_length_count_mat_explicit[i][j] = CONTRA_IL_EXPLICIT_FES[i][j];
      }
    }
    for (v, &w) in self.bulge_loop_0x1_length_counts.iter_mut().zip(CONTRA_BL_0X1_FES.iter()) {
      *v = w;
    }
    let len = self.interior_loop_1x1_length_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        self.interior_loop_1x1_length_count_mat[i][j] = CONTRA_IL_1X1_FES[i][j];
      }
    }
    self.multi_loop_base_count = CONTRA_ML_BASE_FE;
    self.multi_loop_basepairing_count = CONTRA_ML_PAIRED_FE;
    self.multi_loop_accessible_baseunpairing_count = CONTRA_ML_UNPAIRED_FE;
    self.external_loop_accessible_basepairing_count = CONTRA_EL_PAIRED_FE;
    self.external_loop_accessible_baseunpairing_count = CONTRA_EL_UNPAIRED_FE;
    let len = self.basepair_align_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        if !is_canonical(&(i, j)) {continue;}
        for k in 0 .. len {
          for l in 0 .. len {
            if !is_canonical(&(k, l)) {continue;}
            self.basepair_align_count_mat[i][j][k][l] = RIBOSUM_BPA_SCORE_MAT[&((i, j), (k, l))];
          }
        }
      }
    }
    let len = self.loop_align_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        self.loop_align_count_mat[i][j] = RIBOSUM_BA_SCORE_MAT[&(i, j)];
      }
    }
    self.match_2_match_count = MATCH_2_MATCH_SCORE;
    self.match_2_insert_count = MATCH_2_INSERT_SCORE;
    self.insert_extend_count = INSERT_EXTEND_SCORE;
    self.insert_switch_count = INSERT_SWITCH_SCORE;
    let len = self.insert_counts.len();
    for i in 0 .. len {
      self.insert_counts[i] = INSERT_SCORES[i];
    }
    let len = self.align_count_mat.len();
    for i in 0 .. len {
      for j in 0 .. len {
        self.align_count_mat[i][j] = BA_SCORE_MAT[&(i, j)];
      }
    }
    self.accumulate();
  }
}

impl<T: Hash + ToPrimitive + Clone> StaProbMats<T> {
  pub fn origin() -> StaProbMats<T> {
    let prob_mat_pair = (SparseProbMat::<T>::default(), SparseProbMat::<T>::default());
    let prob_set_pair = (Vec::new(), Vec::new());
    StaProbMats {
      bpp_mat_pair: prob_mat_pair.clone(),
      upp_mat_pair_4_hl: prob_set_pair.clone(),
      upp_mat_pair_4_2l: prob_set_pair.clone(),
      upp_mat_pair_4_ml: prob_set_pair.clone(),
      upp_mat_pair_4_el: prob_set_pair.clone(),
      bpp_mat_pair_2: prob_set_pair,
      basepair_align_prob_mat: Prob4dMat::<T>::default(),
      loop_align_prob_mat: SparseProbMat::<T>::default(),
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
      upp_mat_pair_4_hl: prob_set_pair.clone(),
      upp_mat_pair_4_2l: prob_set_pair.clone(),
      upp_mat_pair_4_ml: prob_set_pair.clone(),
      upp_mat_pair_4_el: prob_set_pair.clone(),
      bpp_mat_pair_2: prob_set_pair,
      basepair_align_prob_mat: Prob4dMat::<T>::default(),
      loop_align_prob_mat: SparseProbMat::<T>::default(),
    }
  }
}

impl<T: Hash + Clone> PctStaProbMats<T> {
  pub fn origin() -> PctStaProbMats<T> {
    let prob_mat = SparseProbMat::<T>::default();
    let probs = Vec::new();
    PctStaProbMats {
      bpp_mat: prob_mat.clone(),
      bpp_mat_on_ss: prob_mat,
      upp_mat_4_hl: probs.clone(),
      upp_mat_4_2l: probs.clone(),
      upp_mat_4_ml: probs.clone(),
      upp_mat_4_el: probs.clone(),
      bpp_mat_2: probs,
    }
  }
  pub fn new(seq_len: usize) -> PctStaProbMats<T> {
    let prob_mat = SparseProbMat::<T>::default();
    let probs = vec![0.; seq_len];
    PctStaProbMats {
      bpp_mat: prob_mat.clone(),
      bpp_mat_on_ss: prob_mat,
      upp_mat_4_hl: probs.clone(),
      upp_mat_4_2l: probs.clone(),
      upp_mat_4_ml: probs.clone(),
      upp_mat_4_el: probs.clone(),
      bpp_mat_2: probs,
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
      backward_part_func_set_mat_4_external_loop: part_func_set_mat,
      forward_tmp_part_func_set_mats_with_pos_pairs: tmp_part_func_set_mats_with_pos_pairs.clone(),
      backward_tmp_part_func_set_mats_with_pos_pairs: tmp_part_func_set_mats_with_pos_pairs.clone(),
      forward_2loop_tmp_part_func_set_mats_with_pos_quadruples: tmp_part_func_set_mats_with_pos_quadruples.clone(),
      backward_2loop_tmp_part_func_set_mats_with_pos_quadruples: tmp_part_func_set_mats_with_pos_quadruples.clone(),
    }
  }
}

impl FastaRecord {
  pub fn origin() -> FastaRecord {
    FastaRecord {
      fasta_id: FastaId::new(),
      seq: Seq::new(),
    }
  }
  pub fn new(fasta_id: FastaId, seq: Seq) -> FastaRecord {
    FastaRecord {
      fasta_id: fasta_id,
      seq: seq,
    }
  }
}

impl<T: Hash + Clone + Unsigned + PrimInt + FromPrimitive + Integer + Ord + Sync + Send + Display> TrainDatum<T> {
  pub fn origin() -> TrainDatum<T> {
    TrainDatum {
      seq_pair: (Seq::new(), Seq::new()),
      observed_feature_count_sets: FeatureCountSets::new(0.),
      expected_feature_count_sets: FeatureCountSets::new(NEG_INFINITY),
      bpp_mat_pair: (SparseProbMat::<T>::default(), SparseProbMat::<T>::default()),
      max_bp_span_pair: (T::zero(), T::zero()),
      part_func: NEG_INFINITY,
      forward_pos_pair_mat_set: PosPairMatSet::<T>::default(),
      backward_pos_pair_mat_set: PosPairMatSet::<T>::default(),
      pos_quadruple_mat: PosQuadrupleMat::<T>::default(),
      bp_score_param_set_pair: (BpScoreParamSets::<T>::new(), BpScoreParamSets::<T>::new()),
    }
  }

  pub fn new(input_file_path: &Path, min_bpp: Prob, offset_4_max_gap_num: T) -> TrainDatum<T> {
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
      remove_small_bpps_from_bpp_mat::<T>(&mccaskill_algo(&seq_pair_without_gaps.0[..], false).0, min_bpp),
      remove_small_bpps_from_bpp_mat::<T>(&mccaskill_algo(&seq_pair_without_gaps.1[..], false).0, min_bpp),
    );
    let seq_len_pair = (T::from_usize(seq_pair_without_gaps.0.len()).unwrap(), T::from_usize(seq_pair_without_gaps.1.len()).unwrap());
    let max_gap_num = offset_4_max_gap_num
      + max(seq_len_pair.0, seq_len_pair.1) - min(seq_len_pair.0, seq_len_pair.1);
    let max_gap_num_4_il = max(
      min(max_gap_num, T::from_usize(MAX_GAP_NUM_4_IL).unwrap()),
      T::from_usize(MIN_GAP_NUM_4_IL).unwrap(),
    );
    let pseudo_pos_quadruple = (
      T::zero(),
      seq_len_pair.0 + T::one(),
      T::zero(),
      seq_len_pair.1 + T::one(),
    );
    let (forward_pos_pair_mat_set, backward_pos_pair_mat_set, pos_quadruple_mat) = get_sparse_pos_sets(&(&bpp_mat_pair.0, &bpp_mat_pair.1), max_gap_num_4_il, &pseudo_pos_quadruple);
    let max_bp_span_pair = (
      get_max_bp_span::<T>(&bpp_mat_pair.0),
      get_max_bp_span::<T>(&bpp_mat_pair.1),
    );
    let mut train_datum = TrainDatum {
      seq_pair: seq_pair_without_gaps,
      observed_feature_count_sets: FeatureCountSets::new(0.),
      expected_feature_count_sets: FeatureCountSets::new(NEG_INFINITY),
      bpp_mat_pair: bpp_mat_pair,
      max_bp_span_pair: max_bp_span_pair,
      part_func: NEG_INFINITY,
      forward_pos_pair_mat_set: forward_pos_pair_mat_set,
      backward_pos_pair_mat_set: backward_pos_pair_mat_set,
      pos_quadruple_mat: pos_quadruple_mat,
      bp_score_param_set_pair: (BpScoreParamSets::<T>::new(), BpScoreParamSets::<T>::new()),
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
        let dict_min_align = get_dict_min_align(&char_pair);
        self.observed_feature_count_sets.align_count_mat[dict_min_align.0][dict_min_align.1] += 1.;
        if is_inserting || is_inserting_2 {
          self.observed_feature_count_sets.match_2_insert_count += 1.;
        } else {
          self.observed_feature_count_sets.match_2_match_count += 1.;
        }
        is_inserting = false;
        is_inserting_2 = is_inserting;
        continue;
      }
      if char_pair.1 == PSEUDO_BASE {
        if is_inserting {
          self.observed_feature_count_sets.insert_extend_count += 1.;
        } else if is_inserting_2 {
          self.observed_feature_count_sets.insert_switch_count += 1.;
          is_inserting = true;
          is_inserting_2 = false;
        } else {
          self.observed_feature_count_sets.match_2_insert_count += 1.;
          is_inserting = true;
        }
        self.observed_feature_count_sets.insert_counts[char_pair.0] += 1.;
      } else if char_pair.0 == PSEUDO_BASE {
        if is_inserting_2 {
          self.observed_feature_count_sets.insert_extend_count += 1.;
        } else if is_inserting {
          self.observed_feature_count_sets.insert_switch_count += 1.;
          is_inserting_2 = true;
          is_inserting = false;
        } else {
          self.observed_feature_count_sets.match_2_insert_count += 1.;
          is_inserting_2 = true;
        }
        self.observed_feature_count_sets.insert_counts[char_pair.1] += 1.;
      } else {
        let dict_min_loop_align = get_dict_min_loop_align(&char_pair);
        self.observed_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1] += 1.;
        self.observed_feature_count_sets.align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1] += 1.;
        if is_inserting || is_inserting_2 {
          self.observed_feature_count_sets.match_2_insert_count += 1.;
        } else {
          self.observed_feature_count_sets.match_2_match_count += 1.;
        }
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
        let dict_min_basepair_align = get_dict_min_basepair_align(&base_pair_1, &base_pair_2);
        self.observed_feature_count_sets.basepair_align_count_mat[dict_min_basepair_align.0.0][dict_min_basepair_align.0.1][dict_min_basepair_align.1.0][dict_min_basepair_align.1.1] += 1.;
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
          let dict_min_stack = get_dict_min_stack(&base_pair, &base_pair_3);
          self.observed_feature_count_sets.stack_count_mat[dict_min_stack.0.0][dict_min_stack.0.1][dict_min_stack.1.0][dict_min_stack.1.1] += 1.;
        } else {
          if twoloop_length_pair.0 == 0 || twoloop_length_pair.1 == 0 {
            if sum <= CONSPROB_MAX_TWOLOOP_LEN {
              self.observed_feature_count_sets.bulge_loop_length_counts[sum - 1] += 1.;
              if sum == 1 {
                let mismatch = if twoloop_length_pair.0 == 0 {mismatch_pair.1} else {mismatch_pair.0};
                self.observed_feature_count_sets.bulge_loop_0x1_length_counts[mismatch] += 1.;
              }
            }
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
          let dict_min_stack_2 = get_dict_min_stack(&base_pair_2, &base_pair_4);
          self.observed_feature_count_sets.stack_count_mat[dict_min_stack_2.0.0][dict_min_stack_2.0.1][dict_min_stack_2.1.0][dict_min_stack_2.1.1] += 1.;
        } else {
          if twoloop_length_pair_2.0 == 0 || twoloop_length_pair_2.1 == 0 {
            if sum_2 <= CONSPROB_MAX_TWOLOOP_LEN {
              self.observed_feature_count_sets.bulge_loop_length_counts[sum_2 - 1] += 1.;
              if sum_2 == 1 {
                let mismatch_2 = if twoloop_length_pair_2.0 == 0 {mismatch_pair_2.1} else {mismatch_pair_2.0};
                self.observed_feature_count_sets.bulge_loop_0x1_length_counts[mismatch_2] += 1.;
              }
            }
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
      self.observed_feature_count_sets.helix_end_count_mat[base_pair.1][base_pair.0] += 1.;
      self.observed_feature_count_sets.helix_end_count_mat[base_pair_2.1][base_pair_2.0] += 1.;
    }
  }

  pub fn set_curr_params(&mut self, feature_score_sets: &FeatureCountSets) {
    self.bp_score_param_set_pair.0 = BpScoreParamSets::<T>::set_curr_params(feature_score_sets, &self.seq_pair.0, &self.bpp_mat_pair.0);
    self.bp_score_param_set_pair.1 = BpScoreParamSets::<T>::set_curr_params(feature_score_sets, &self.seq_pair.1, &self.bpp_mat_pair.1);
  }
}

impl<T: Hash + Clone> AlignProbMats<T> {
  pub fn new() -> AlignProbMats<T> {
    AlignProbMats {
      loop_align_prob_mat: SparseProbMat::<T>::default(),
      basepair_align_prob_mat: Prob4dMat::<T>::default(),
      align_prob_mat: SparseProbMat::<T>::default(),
    }
  }
}

pub const MAX_GAP_NUM_4_IL: usize = 15;
pub const MIN_GAP_NUM_4_IL: usize = 2;
pub const MAX_GAP_NUM_4_IL_TRAIN: usize = MAX_GAP_NUM_4_IL;
pub const MIN_GAP_NUM_4_IL_TRAIN: usize = MIN_GAP_NUM_4_IL;
pub const DEFAULT_MIN_BPP: Prob = 0.04;
pub const DEFAULT_MIN_BPP_4_TRAIN: Prob = DEFAULT_MIN_BPP;
pub const DEFAULT_OFFSET_4_MAX_GAP_NUM: usize = 4;
pub const DEFAULT_OFFSET_4_MAX_GAP_NUM_TRAIN: usize = DEFAULT_OFFSET_4_MAX_GAP_NUM;
pub const NUM_OF_BASES: usize = 4;
pub const NUM_OF_BASEPAIRINGS: usize = 6;
pub const CONSPROB_MAX_HAIRPIN_LOOP_LEN: usize = 30;
pub const CONSPROB_MAX_TWOLOOP_LEN: usize = CONSPROB_MAX_HAIRPIN_LOOP_LEN;
pub const CONSPROB_MIN_HAIRPIN_LOOP_LEN: usize = 3;
pub const CONSPROB_MIN_HAIRPIN_LOOP_SPAN: usize = CONSPROB_MIN_HAIRPIN_LOOP_LEN + 2;
pub const CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT: usize = 4;
pub const CONSPROB_MAX_INTERIOR_LOOP_LEN_SYMM: usize = CONSPROB_MAX_TWOLOOP_LEN / 2;
pub const CONSPROB_MAX_INTERIOR_LOOP_LEN_ASYMM: usize = CONSPROB_MAX_TWOLOOP_LEN - 2;
pub const GAMMA_DIST_ALPHA: FeatureCount = 0.;
pub const GAMMA_DIST_BETA: FeatureCount = 1.;
pub const LEARNING_TOLERANCE: FeatureCount = 0.02;
pub const BPP_MAT_FILE_NAME: &'static str = "bpp_mats.dat";
pub const BPP_MAT_ON_SS_FILE_NAME: &'static str = "bpp_mats_on_ss.dat";
pub const UPP_MAT_ON_HL_FILE_NAME: &'static str = "upp_mats_on_hl.dat";
pub const UPP_MAT_ON_2L_FILE_NAME: &'static str = "upp_mats_on_2l.dat";
pub const UPP_MAT_ON_ML_FILE_NAME: &'static str = "upp_mats_on_ml.dat";
pub const UPP_MAT_ON_EL_FILE_NAME: &'static str = "upp_mats_on_el.dat";
pub const BPP_MAT_FILE_NAME_2: &'static str = "bpp_mats_2.dat";
pub const BASEPAIR_ALIGN_PROB_MAT_FILE_NAME: &'static str = "basepair_align_prob_mat.dat";
pub const LOOP_ALIGN_PROB_MAT_FILE_NAME: &'static str = "loop_align_prob_mat.dat";
pub const ALIGN_PROB_MAT_FILE_NAME: &'static str = "align_prob_mat.dat";
pub const TRAINED_FEATURE_SCORE_SETS_FILE_PATH: &'static str = "../src/trained_feature_score_sets.rs";
pub const README_FILE_NAME: &str = "README.md";
pub const README_CONTENTS: &str = "# bpp_mats.dat\nThis file contains average probabilistic consistency based on posterior nucleotide pair-matching probabilities. You can treat this average probabilistic consistency like conventional nucleotide base-pairing probabilities. Nucleotide positions are indexed starting from zero.\n\n# bpp_mats_on_ss.dat\nThis file contains McCaskill's nucleotide base-pairing probabilities on RNA secondary structures. These nucleotide base-pairing probabilities are used to accelerate ConsProb's inside-outside algorithm.\n\n# bpp_mats_2.dat\nThis file contains average probabilistic consistency per nucleotide. This average probabilistic consistency is obtained by marginalizing one nucleotide for average probabilistic consistency in \"bpp_mats.dat.\"\n\n# upp_mats_on_x.dat\nThis file type contains average probabilistic consistency per nucleotide. This average probabilistic consistency is for nucleotide unpairing and under the structural context \"x.\" \"hl,\" \"2l,\" \"ml,\" \"el\" stand for hairpin loops, 2-loops, multi-loops, external loops, respectively.\n\n# basepair_align_prob_mat.dat\nThis file contains three-way probabilistic consistency based on posterior nucleotide pair-matching probabilities. You can treat this three-way probabilistic consistency like posterior nucleotide pair-matching probabilities.\n\n# loop_align_prob_mat.dat\nThis file contains three-way probabilistic consistency based on posterior nucleotide loop-matching probabilities. You can treat this three-way probabilistic consistency like posterior nucleotide loop-matching probabilities.\n\nalign_prob_mat.dat\nThis file contains three-way probabilistic consistency based on posterior nucleotide matching probabilities. You can treat this three-way probabilistic consistency like posterior nucleotide matching probabilities.";

pub fn io_algo_4_prob_mats<T>(
  seq_pair: &SeqPair,
  feature_score_sets: &FeatureCountSets,
  max_bp_span_pair: &PosPair<T>,
  max_gap_num: T,
  max_gap_num_4_il: T,
  produces_struct_profs: bool,
  trains_score_params: bool,
  expected_feature_count_sets: &mut FeatureCountSets,
  forward_pos_pair_mat_set: &PosPairMatSet<T>,
  backward_pos_pair_mat_set: &PosPairMatSet<T>,
  pos_quadruple_mat: &PosQuadrupleMat<T>,
  bp_score_param_set_pair: &BpScoreParamSetPair<T>,
  produces_align_probs: bool,
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
    trains_score_params,
    forward_pos_pair_mat_set,
    backward_pos_pair_mat_set,
    pos_quadruple_mat,
    bp_score_param_set_pair,
    produces_align_probs,
  );
  (get_sta_prob_mats::<T>(
    seq_pair,
    feature_score_sets,
    max_bp_span_pair,
    max_gap_num,
    max_gap_num_4_il,
    &sta_part_func_mats,
    produces_struct_profs,
    global_part_func,
    trains_score_params,
    expected_feature_count_sets,
    pos_quadruple_mat,
    bp_score_param_set_pair,
    produces_align_probs,
  ), global_part_func)
}

pub fn get_sta_inside_part_func_mats<T>(
  seq_pair: &SeqPair,
  feature_score_sets: &FeatureCountSets,
  max_bp_span_pair: &PosPair<T>,
  max_gap_num: T,
  max_gap_num_4_il: T,
  trains_score_params: bool,
  forward_pos_pair_mat_set: &PosPairMatSet<T>,
  backward_pos_pair_mat_set: &PosPairMatSet<T>,
  pos_quadruple_mat: &PosQuadrupleMat<T>,
  bp_score_param_set_pair: &BpScoreParamSetPair<T>,
  produces_align_probs: bool,
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
    for substr_len_2 in range_inclusive(
      T::from_usize(CONSPROB_MIN_HAIRPIN_LOOP_SPAN).unwrap(),
      max_bp_span_pair.1,
    ) {
      for &(i, j, k, l) in pos_quadruple_mat {
        if j - i + T::one() != substr_len_1 || l - k + T::one() != substr_len_2 {continue;}
        let (long_i, long_j, long_k, long_l) = (i.to_usize().unwrap(), j.to_usize().unwrap(), k.to_usize().unwrap(), l.to_usize().unwrap());
        let base_pair = (seq_pair.0[long_i], seq_pair.0[long_j]);
        let base_pair_2 = (seq_pair.1[long_k], seq_pair.1[long_l]);
        let pos_quadruple = (i, j, k, l);
        let (forward_tmp_part_func_set_mat, part_func_on_sa, part_func_4_ml, forward_tmp_part_func_set_mat_4_2loop) =
          get_tmp_part_func_set_mat::<T>(
            &seq_pair,
            feature_score_sets,
            max_gap_num_4_il,
            &pos_quadruple,
            &sta_part_func_mats,
            true,
            trains_score_params,
            forward_pos_pair_mat_set,
            backward_pos_pair_mat_set,
            bp_score_param_set_pair,
            produces_align_probs,
          );
        let (backward_tmp_part_func_set_mat, _, _, backward_tmp_part_func_set_mat_4_2loop) = get_tmp_part_func_set_mat::<T>(
          &seq_pair,
          feature_score_sets,
          max_gap_num_4_il,
          &pos_quadruple,
          &sta_part_func_mats,
          false,
          trains_score_params,
          forward_pos_pair_mat_set,
          backward_pos_pair_mat_set,
          bp_score_param_set_pair,
          produces_align_probs,
        );
        let mut sum = NEG_INFINITY;
        let basepair_align_score = feature_score_sets.basepair_align_count_mat[base_pair.0][base_pair.1][base_pair_2.0][base_pair_2.1]
           + feature_score_sets.align_count_mat[base_pair.0][base_pair_2.0]
           + feature_score_sets.align_count_mat[base_pair.1][base_pair_2.1];
        if substr_len_1.to_usize().unwrap() - 2 <= CONSPROB_MAX_HAIRPIN_LOOP_LEN && substr_len_2.to_usize().unwrap() - 2 <= CONSPROB_MAX_HAIRPIN_LOOP_LEN {
          let hairpin_loop_score = bp_score_param_set_pair.0.hairpin_loop_scores[&(i, j)];
          let hairpin_loop_score_2 = bp_score_param_set_pair.1.hairpin_loop_scores[&(k, l)];
          let score =
            basepair_align_score + hairpin_loop_score + hairpin_loop_score_2 + part_func_on_sa;
          logsumexp(&mut sum, score);
        }
        for &(m, n, o, p) in pos_quadruple_mat {
          if !(i < m && n < j) || !(k < o && p < l) {
            continue;
          }
          if m - i - T::one() + j - n - T::one() > T::from_usize(CONSPROB_MAX_TWOLOOP_LEN).unwrap() {
            continue;
          }
          if o - k - T::one() + l - p - T::one() > T::from_usize(CONSPROB_MAX_TWOLOOP_LEN).unwrap() {
            continue;
          }
          let pos_quadruple_2 = (m, n, o, p);
          match sta_part_func_mats
            .part_func_4d_mat_4_bpas
            .get(&pos_quadruple_2)
          {
            Some(&part_func) => {
              let mut forward_term = NEG_INFINITY;
              match forward_tmp_part_func_set_mat.get(&(m - T::one(), o - T::one())) {
                Some(part_func_sets) => {
                  let ref part_funcs = part_func_sets.part_funcs_on_sa;
                  let term = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
                  logsumexp(&mut forward_term, term);
                  let term = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut forward_term, term);
                  let term = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut forward_term, term);
                }, None => {},
              }
              let mut backward_term = NEG_INFINITY;
              match backward_tmp_part_func_set_mat.get(&(n + T::one(), p + T::one())) {
                Some(part_func_sets) => {
                  let ref part_funcs = part_func_sets.part_funcs_on_sa;
                  let term = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
                  logsumexp(&mut backward_term, term);
                  let term = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut backward_term, term);
                  let term = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut backward_term, term);
                }, None => {},
              }
              let part_func_4_2l = forward_term + backward_term;
              let twoloop_score = bp_score_param_set_pair.0.twoloop_scores[&(i, j, m, n)];
              let twoloop_score_2 = bp_score_param_set_pair.1.twoloop_scores[&(k, l, o, p)];
              let coefficient =
                basepair_align_score + twoloop_score + twoloop_score_2 + part_func;
              logsumexp(&mut sum, coefficient + part_func_4_2l);
            }
            None => {}
          }
        }
        let multi_loop_closing_basepairing_score = bp_score_param_set_pair.0.multi_loop_closing_bp_scores[&(i, j)];
        let multi_loop_closing_basepairing_score_2 = bp_score_param_set_pair.1.multi_loop_closing_bp_scores[&(k, l)];
        let score = basepair_align_score
          + multi_loop_closing_basepairing_score
          + multi_loop_closing_basepairing_score_2
          + part_func_4_ml;
        logsumexp(&mut sum, score);
        if sum > NEG_INFINITY {
          sta_part_func_mats
            .part_func_4d_mat_4_bpas
            .insert(pos_quadruple, sum);
          let multi_loop_accessible_basepairing_score = bp_score_param_set_pair.0.multi_loop_accessible_bp_scores[&(i, j)];
          let multi_loop_accessible_basepairing_score_2 = bp_score_param_set_pair.1.multi_loop_accessible_bp_scores[&(k, l)];
          let external_loop_accessible_basepairing_score = bp_score_param_set_pair.0.external_loop_accessible_bp_scores[&(i, j)];
          let external_loop_accessible_basepairing_score_2 = bp_score_param_set_pair.1.external_loop_accessible_bp_scores[&(k, l)];
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
        if produces_align_probs || trains_score_params {
          sta_part_func_mats
            .forward_2loop_tmp_part_func_set_mats_with_pos_quadruples
            .insert(pos_quadruple, forward_tmp_part_func_set_mat_4_2loop);
          sta_part_func_mats
            .backward_2loop_tmp_part_func_set_mats_with_pos_quadruples
            .insert(pos_quadruple, backward_tmp_part_func_set_mat_4_2loop);
        }
      }
    }
  }
  let leftmost_pos_pair = (T::zero(), T::zero());
  let rightmost_pos_pair = (seq_len_pair.0 - T::one(), seq_len_pair.1 - T::one());
  let mut part_funcs = TmpPartFuncs::new();
  part_funcs.part_func_4_align = 0.;
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
      match backward_pos_pair_mat_set.get(&pos_pair) {
        Some(backward_pos_pair_mat) => {
          for &(k, l) in backward_pos_pair_mat {
            let pos_pair_2 = (k - T::one(), l - T::one());
            let pos_quadruple = (k, i, l, j);
            let is_begin = pos_pair_2 == leftmost_pos_pair;
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
                    let score = part_funcs.part_func_4_align + if is_begin {0.} else {feature_score_sets.match_2_match_count} + part_func;
                    logsumexp(&mut sum, score);
                    let score = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + part_func;
                    logsumexp(&mut sum, score);
                    let score = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + part_func;
                    logsumexp(&mut sum, score);
                  }
                  None => {}
                }
              }
              None => {}
            }
          }
        }, None => {},
      }
      let base_2 = seq_pair.1[long_j];
      if i > T::zero() && j > T::zero() {
        let loop_align_score = feature_score_sets.loop_align_count_mat[base][base_2] + feature_score_sets.align_count_mat[base][base_2] + 2. * feature_score_sets.external_loop_accessible_baseunpairing_count;
        let pos_pair_2 = (i - T::one(), j - T::one());
        let is_begin = pos_pair_2 == leftmost_pos_pair;
        match sta_part_func_mats
          .forward_part_func_set_mat_4_external_loop
          .get(&pos_pair_2)
        {
          Some(part_funcs) => {
            let score = part_funcs.part_func_4_align + if is_begin {0.} else {feature_score_sets.match_2_match_count} + loop_align_score;
            logsumexp(&mut sum, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + loop_align_score;
            logsumexp(&mut sum, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + loop_align_score;
            logsumexp(&mut sum, score);
          }
          None => {}
        }
        part_funcs.part_func_4_align = sum;
      }
      if i > T::zero() {
        let insert_score = feature_score_sets.insert_counts[base];
        sum = NEG_INFINITY;
        let pos_pair_2 = (i - T::one(), j);
        let is_begin = pos_pair_2 == leftmost_pos_pair;
        match sta_part_func_mats
          .forward_part_func_set_mat_4_external_loop
          .get(&pos_pair_2)
        {
          Some(part_funcs) => {
            let score = part_funcs.part_func_4_align + if is_begin {0.} else {feature_score_sets.match_2_insert_count};
            logsumexp(&mut sum, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.insert_extend_count;
            logsumexp(&mut sum, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_switch_count;
            logsumexp(&mut sum, score);
          }
          None => {}
        }
        sum += insert_score + feature_score_sets.external_loop_accessible_baseunpairing_count;
        part_funcs.part_func_4_insert = sum;
      }
      if j > T::zero() {
        let insert_score = feature_score_sets.insert_counts[base_2];
        sum = NEG_INFINITY;
        let pos_pair_2 = (i, j - T::one());
        let is_begin = pos_pair_2 == leftmost_pos_pair;
        match sta_part_func_mats
          .forward_part_func_set_mat_4_external_loop
          .get(&pos_pair_2)
        {
          Some(part_funcs) => {
            let score = part_funcs.part_func_4_align + if is_begin {0.} else {feature_score_sets.match_2_insert_count};
            logsumexp(&mut sum, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.insert_switch_count;
            logsumexp(&mut sum, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_extend_count;
            logsumexp(&mut sum, score);
          }
          None => {}
        }
        sum += insert_score + feature_score_sets.external_loop_accessible_baseunpairing_count;
        part_funcs.part_func_4_insert_2 = sum;
      }
      sta_part_func_mats
        .forward_part_func_set_mat_4_external_loop
        .insert(pos_pair, part_funcs);
    }
  }
  let ref part_funcs = sta_part_func_mats.forward_part_func_set_mat_4_external_loop[&(
    seq_len_pair.0 - T::from_usize(2).unwrap(),
    seq_len_pair.1 - T::from_usize(2).unwrap(),
  )];
  let mut global_part_func = part_funcs.part_func_4_align;
  logsumexp(&mut global_part_func, part_funcs.part_func_4_insert);
  logsumexp(&mut global_part_func, part_funcs.part_func_4_insert_2);
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
      match forward_pos_pair_mat_set.get(&pos_pair) {
        Some(forward_pos_pair_mat) => {
          for &(k, l) in forward_pos_pair_mat {
            let pos_pair_2 = (k + T::one(), l + T::one());
            let is_end = pos_pair_2 == rightmost_pos_pair;
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
                    let score = part_funcs.part_func_4_align + if is_end {0.} else {feature_score_sets.match_2_match_count} + part_func;
                    logsumexp(&mut sum, score);
                    let score = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + part_func;
                    logsumexp(&mut sum, score);
                    let score = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + part_func;
                    logsumexp(&mut sum, score);
                  }
                  None => {}
                }
              }
              None => {}
            }
          }
        }, None => {},
      }
      let base_2 = seq_pair.1[long_j];
      if i < seq_len_pair.0 - T::one() && j < seq_len_pair.1 - T::one() {
        let pos_pair_2 = (i + T::one(), j + T::one());
        let is_end = pos_pair_2 == rightmost_pos_pair;
        let loop_align_score = feature_score_sets.loop_align_count_mat[base][base_2] + feature_score_sets.align_count_mat[base][base_2] + 2. * feature_score_sets.external_loop_accessible_baseunpairing_count;
        match sta_part_func_mats
          .backward_part_func_set_mat_4_external_loop
          .get(&pos_pair_2)
        {
          Some(part_funcs) => {
            let score = part_funcs.part_func_4_align + if is_end {0.} else {feature_score_sets.match_2_match_count} + loop_align_score;
            logsumexp(&mut sum, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + loop_align_score;
            logsumexp(&mut sum, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + loop_align_score;
            logsumexp(&mut sum, score);
          }
          None => {}
        }
        part_funcs.part_func_4_align = sum;
      }
      if i < seq_len_pair.0 - T::one() {
        let insert_score = feature_score_sets.insert_counts[base];
        sum = NEG_INFINITY;
        let pos_pair_2 = (i + T::one(), j);
        let is_end = pos_pair_2 == rightmost_pos_pair;
        match sta_part_func_mats
          .backward_part_func_set_mat_4_external_loop
          .get(&pos_pair_2)
        {
          Some(part_funcs) => {
            let score = part_funcs.part_func_4_align + if is_end {0.} else {feature_score_sets.match_2_insert_count};
            logsumexp(&mut sum, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.insert_extend_count;
            logsumexp(&mut sum, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_switch_count;
            logsumexp(&mut sum, score);
          }
          None => {}
        }
        sum += insert_score + feature_score_sets.external_loop_accessible_baseunpairing_count;
        part_funcs.part_func_4_insert = sum;
      }
      if j < seq_len_pair.1 - T::one() {
        let insert_score = feature_score_sets.insert_counts[base_2];
        sum = NEG_INFINITY;
        let pos_pair_2 = (i, j + T::one());
        let is_end = pos_pair_2 == rightmost_pos_pair;
        match sta_part_func_mats
          .backward_part_func_set_mat_4_external_loop
          .get(&pos_pair_2)
        {
          Some(part_funcs) => {
            let score = part_funcs.part_func_4_align + if is_end {0.} else {feature_score_sets.match_2_insert_count};
            logsumexp(&mut sum, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.insert_switch_count;
            logsumexp(&mut sum, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_extend_count;
            logsumexp(&mut sum, score);
          }
          None => {}
        }
        sum += insert_score + feature_score_sets.external_loop_accessible_baseunpairing_count;
        part_funcs.part_func_4_insert_2 = sum;
      }
      sta_part_func_mats
        .backward_part_func_set_mat_4_external_loop
        .insert(pos_pair, part_funcs);
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
  is_forward: bool,
  trains_score_params: bool,
  forward_pos_pair_mat_set: &PosPairMatSet<T>,
  backward_pos_pair_mat_set: &PosPairMatSet<T>,
  bp_score_param_set_pair: &BpScoreParamSetPair<T>,
  produces_align_probs: bool,
) -> (TmpPartFuncSetMat<T>, PartFunc, PartFunc, PartFuncSetMat<T>)
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
  let pos_pair_mat_set = if is_forward {
    backward_pos_pair_mat_set
  } else {
    forward_pos_pair_mat_set
  };
  let needs_twoloop_part_funcs = produces_align_probs || trains_score_params;
  for &u in iter.iter() {
    let long_u = u.to_usize().unwrap();
    let base = seq_pair.0[long_u];
    let insert_score = feature_score_sets.insert_counts[base];
    for &v in iter_2.iter() {
      let pos_pair = (u, v);
      if !is_min_gap_ok(&pos_pair, &pseudo_pos_quadruple, max_gap_num_4_il) {
        continue;
      }
      let skips = cache_is_used && tmp_part_func_set_mat.contains_key(&pos_pair);
      if skips && !needs_twoloop_part_funcs {
        continue;
      }
      let mut tmp_part_func_sets = TmpPartFuncSets::new();
      let mut tmp_part_funcs_4_2loop = TmpPartFuncs::new();
      if (is_forward && u == i && v == k) || (!is_forward && u == j && v == l) {
        tmp_part_func_sets.part_funcs_on_sa.part_func_4_align = 0.;
        tmp_part_func_sets.part_funcs_on_sa_4_ml.part_func_4_align = 0.;
        tmp_part_func_sets.part_funcs_on_mls.part_func_4_align = 0.;
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
      match pos_pair_mat_set.get(&pos_pair) {
        Some(pos_pair_mat) => {
          for &(m, n) in pos_pair_mat {
            if is_forward {
              if !(i < m && k < n) {continue;}
            } else {
              if !(m < j && n < l) {continue;}
            }
            let pos_pair_2 = if is_forward {
              (m - T::one(), n - T::one())
            } else {
              (m + T::one(), n + T::one())
            };
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
                    let score = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count + part_func;
                    logsumexp(&mut sum_4_ml, score);
                    let score = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + part_func;
                    logsumexp(&mut sum_4_ml, score);
                    let score = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + part_func;
                    logsumexp(&mut sum_4_ml, score);
                    let ref part_funcs = part_func_sets.part_funcs_on_sa_4_ml;
                    let score = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count + part_func;
                    logsumexp(&mut sum_4_first_bpas_on_mls, score);
                    let score = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + part_func;
                    logsumexp(&mut sum_4_first_bpas_on_mls, score);
                    let score = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + part_func;
                    logsumexp(&mut sum_4_first_bpas_on_mls, score);
                  }
                  if needs_twoloop_part_funcs {
                    if pos_quadruple_2.0 - i - T::one() + j - pos_quadruple_2.1 - T::one() > T::from_usize(CONSPROB_MAX_TWOLOOP_LEN).unwrap() {
                      continue;
                    }
                    if pos_quadruple_2.2 - k - T::one() + l - pos_quadruple_2.3 - T::one() > T::from_usize(CONSPROB_MAX_TWOLOOP_LEN).unwrap() {
                      continue;
                    }
                    let part_func = sta_part_func_mats
                      .part_func_4d_mat_4_bpas[&pos_quadruple_2];
                    let twoloop_score = bp_score_param_set_pair.0.twoloop_scores[&(i, j, pos_quadruple_2.0, pos_quadruple_2.1)];
                    let twoloop_score_2 = bp_score_param_set_pair.1.twoloop_scores[&(k, l, pos_quadruple_2.2, pos_quadruple_2.3)];
                    let ref part_funcs = part_func_sets.part_funcs_on_sa;
                    let score = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count + part_func + twoloop_score + twoloop_score_2;
                    logsumexp(&mut sum_4_2loop, score);
                    let score = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + part_func + twoloop_score + twoloop_score_2;
                    logsumexp(&mut sum_4_2loop, score);
                    let score = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + part_func + twoloop_score + twoloop_score_2;
                    logsumexp(&mut sum_4_2loop, score);
                  }
                }
                None => {}
              },
              None => {}
            }
          }
        }, None => {},
      }
      let pos_pair_2 = if is_forward {
        (u - T::one(), v - T::one())
      } else {
        (u + T::one(), v + T::one())
      };
      let base_2 = seq_pair.1[long_v];
      let loop_align_score = feature_score_sets.loop_align_count_mat[base][base_2] + feature_score_sets.align_count_mat[base][base_2];
      match tmp_part_func_set_mat.get(&pos_pair_2) {
        Some(part_func_sets) => {
          if !skips {
            let ref part_funcs = part_func_sets.part_funcs_4_ml;
            let score = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count + loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
            logsumexp(&mut sum_4_ml, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
            logsumexp(&mut sum_4_ml, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
            logsumexp(&mut sum_4_ml, score);
            let ref part_funcs = part_func_sets.part_funcs_4_first_bpas_on_mls;
            let score = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count + loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
            logsumexp(&mut sum_4_first_bpas_on_mls, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
            logsumexp(&mut sum_4_first_bpas_on_mls, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
            logsumexp(&mut sum_4_first_bpas_on_mls, score);
            let ref part_funcs = part_func_sets.part_funcs_on_sa_4_ml;
            let score = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count + loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
            logsumexp(&mut sum_on_sa_4_ml, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
            logsumexp(&mut sum_on_sa_4_ml, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
            logsumexp(&mut sum_on_sa_4_ml, score);
            let ref part_funcs = part_func_sets.part_funcs_on_sa;
            let score = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count + loop_align_score;
            logsumexp(&mut sum_on_sa, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + loop_align_score;
            logsumexp(&mut sum_on_sa, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + loop_align_score;
            logsumexp(&mut sum_on_sa, score);
          }
          if needs_twoloop_part_funcs {
            match tmp_part_func_set_mat_4_2loop.get(&pos_pair_2) {
              Some(part_funcs) => {
                let score = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count + loop_align_score;
                logsumexp(&mut sum_4_2loop, score);
                let score = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + loop_align_score;
                logsumexp(&mut sum_4_2loop, score);
                let score = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + loop_align_score;
                logsumexp(&mut sum_4_2loop, score);
              }, None => {},
            }
          }
        }
        None => {}
      }
      if !skips {
        tmp_part_func_sets.part_funcs_4_ml.part_func_4_align = sum_4_ml;
        logsumexp(&mut tmp_sum, sum_4_ml);
        tmp_part_func_sets
          .part_funcs_4_first_bpas_on_mls
          .part_func_4_align = sum_4_first_bpas_on_mls;
        logsumexp(&mut tmp_sum, sum_4_first_bpas_on_mls);
        tmp_part_func_sets
          .part_funcs_4_bpas_on_mls
          .part_func_4_align = tmp_sum;
          tmp_part_func_sets.part_funcs_on_sa_4_ml.part_func_4_align = sum_on_sa_4_ml;
          logsumexp(&mut tmp_sum, sum_on_sa_4_ml);
          tmp_part_func_sets.part_funcs_on_mls.part_func_4_align = tmp_sum;
          tmp_part_func_sets.part_funcs_on_sa.part_func_4_align = sum_on_sa;
      }
      if needs_twoloop_part_funcs {
        tmp_part_funcs_4_2loop.part_func_4_align = sum_4_2loop;
      }
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
            let score = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
            logsumexp(&mut sum_4_ml, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.insert_extend_count;
            logsumexp(&mut sum_4_ml, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_switch_count;
            logsumexp(&mut sum_4_ml, score);
            let ref part_funcs = part_func_sets.part_funcs_4_first_bpas_on_mls;
            let score = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
            logsumexp(&mut sum_4_first_bpas_on_mls, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.insert_extend_count;
            logsumexp(&mut sum_4_first_bpas_on_mls, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_switch_count;
            logsumexp(&mut sum_4_first_bpas_on_mls, score);
            let ref part_funcs = part_func_sets.part_funcs_on_sa_4_ml;
            let score = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
            logsumexp(&mut sum_on_sa_4_ml, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.insert_extend_count;
            logsumexp(&mut sum_on_sa_4_ml, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_switch_count;
            logsumexp(&mut sum_on_sa_4_ml, score);
            let ref part_funcs = part_func_sets.part_funcs_on_sa;
            let score = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
            logsumexp(&mut sum_on_sa, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.insert_extend_count;
            logsumexp(&mut sum_on_sa, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_switch_count;
            logsumexp(&mut sum_on_sa, score);
          }
          if needs_twoloop_part_funcs {
            match tmp_part_func_set_mat_4_2loop.get(&pos_pair_2) {
              Some(part_funcs) => {
                let score = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
                logsumexp(&mut sum_4_2loop, score);
                let score = part_funcs.part_func_4_insert + feature_score_sets.insert_extend_count;
                logsumexp(&mut sum_4_2loop, score);
                let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_switch_count;
                logsumexp(&mut sum_4_2loop, score);
              }, None => {},
            }
          }
        }
        None => {}
      }
      if !skips {
        sum_4_ml += feature_score_sets.multi_loop_accessible_baseunpairing_count + insert_score;
        sum_4_first_bpas_on_mls += feature_score_sets.multi_loop_accessible_baseunpairing_count + insert_score;
        sum_on_sa_4_ml += feature_score_sets.multi_loop_accessible_baseunpairing_count + insert_score;
        sum_on_sa += insert_score;
        tmp_part_func_sets.part_funcs_4_ml.part_func_4_insert = sum_4_ml;
        logsumexp(&mut tmp_sum, sum_4_ml);
        tmp_part_func_sets
          .part_funcs_4_first_bpas_on_mls
          .part_func_4_insert = sum_4_first_bpas_on_mls;
        logsumexp(&mut tmp_sum, sum_4_first_bpas_on_mls);
        tmp_part_func_sets
          .part_funcs_4_bpas_on_mls
          .part_func_4_insert = tmp_sum;
        tmp_part_func_sets.part_funcs_on_sa_4_ml.part_func_4_insert = sum_on_sa_4_ml;
        logsumexp(&mut tmp_sum, sum_on_sa_4_ml);
        tmp_part_func_sets.part_funcs_on_mls.part_func_4_insert = tmp_sum;
        tmp_part_func_sets.part_funcs_on_sa.part_func_4_insert = sum_on_sa;
      }
      if needs_twoloop_part_funcs {
        tmp_part_funcs_4_2loop.part_func_4_insert = sum_4_2loop + insert_score;
      }
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
            let score = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
            logsumexp(&mut sum_4_ml, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.insert_switch_count;
            logsumexp(&mut sum_4_ml, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_extend_count;
            logsumexp(&mut sum_4_ml, score);
            let ref part_funcs = part_func_sets.part_funcs_4_first_bpas_on_mls;
            let score = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
            logsumexp(&mut sum_4_first_bpas_on_mls, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.insert_switch_count;
            logsumexp(&mut sum_4_first_bpas_on_mls, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_extend_count;
            logsumexp(&mut sum_4_first_bpas_on_mls, score);
            let ref part_funcs = part_func_sets.part_funcs_on_sa_4_ml;
            let score = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
            logsumexp(&mut sum_on_sa_4_ml, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.insert_switch_count;
            logsumexp(&mut sum_on_sa_4_ml, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_extend_count;
            logsumexp(&mut sum_on_sa_4_ml, score);
            let ref part_funcs = part_func_sets.part_funcs_on_sa;
            let score = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
            logsumexp(&mut sum_on_sa, score);
            let score = part_funcs.part_func_4_insert + feature_score_sets.insert_switch_count;
            logsumexp(&mut sum_on_sa, score);
            let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_extend_count;
            logsumexp(&mut sum_on_sa, score);
          }
          if needs_twoloop_part_funcs {
            match tmp_part_func_set_mat_4_2loop.get(&pos_pair_2) {
              Some(part_funcs) => {
                let score = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
                logsumexp(&mut sum_4_2loop, score);
                let score = part_funcs.part_func_4_insert + feature_score_sets.insert_switch_count;
                logsumexp(&mut sum_4_2loop, score);
                let score = part_funcs.part_func_4_insert_2 + feature_score_sets.insert_extend_count;
                logsumexp(&mut sum_4_2loop, score);
              }, None => {},
            }
          }
        }
        None => {}
      }
      let insert_score = feature_score_sets.insert_counts[base_2];
      if !skips {
        sum_4_ml += feature_score_sets.multi_loop_accessible_baseunpairing_count + insert_score;
        sum_4_first_bpas_on_mls += feature_score_sets.multi_loop_accessible_baseunpairing_count + insert_score;
        sum_on_sa_4_ml += feature_score_sets.multi_loop_accessible_baseunpairing_count + insert_score;
        sum_on_sa += insert_score;
        tmp_part_func_sets.part_funcs_4_ml.part_func_4_insert_2 = sum_4_ml;
        logsumexp(&mut tmp_sum, sum_4_ml);
        tmp_part_func_sets
          .part_funcs_4_first_bpas_on_mls
          .part_func_4_insert_2 = sum_4_first_bpas_on_mls;
        logsumexp(&mut tmp_sum, sum_4_first_bpas_on_mls);
        tmp_part_func_sets
          .part_funcs_4_bpas_on_mls
          .part_func_4_insert_2 = tmp_sum;
          tmp_part_func_sets.part_funcs_on_sa_4_ml.part_func_4_insert_2 = sum_on_sa_4_ml;
          logsumexp(&mut tmp_sum, sum_on_sa_4_ml);
          tmp_part_func_sets.part_funcs_on_mls.part_func_4_insert_2 = tmp_sum;
          tmp_part_func_sets.part_funcs_on_sa.part_func_4_insert_2 = sum_on_sa;
      }
      if needs_twoloop_part_funcs {
        tmp_part_funcs_4_2loop.part_func_4_insert_2 = sum_4_2loop + insert_score;
      }
      if !skips {
        tmp_part_func_set_mat.insert(pos_pair, tmp_part_func_sets);
      }
      if needs_twoloop_part_funcs {
        tmp_part_func_set_mat_4_2loop.insert(pos_pair, tmp_part_funcs_4_2loop);
      }
    }
  }
  let ref tmp_part_funcs = tmp_part_func_set_mat[&if is_forward {
    rightmost_pos_pair
  } else {
    leftmost_pos_pair
  }];
  let ref part_funcs = tmp_part_funcs.part_funcs_on_sa;
  let mut edge_part_func_on_sa = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
  logsumexp(&mut edge_part_func_on_sa, part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count);
  logsumexp(&mut edge_part_func_on_sa, part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count);
  let ref part_funcs = tmp_part_funcs.part_funcs_4_ml;
  let mut edge_part_func_4_ml = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
  logsumexp(&mut edge_part_func_4_ml, part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count);
  logsumexp(&mut edge_part_func_4_ml, part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count);
  (
    tmp_part_func_set_mat,
    edge_part_func_on_sa,
    edge_part_func_4_ml,
    tmp_part_func_set_mat_4_2loop,
  )
}

pub fn get_sta_prob_mats<T>(
  seq_pair: &SeqPair,
  feature_score_sets: &FeatureCountSets,
  max_bp_span_pair: &PosPair<T>,
  max_gap_num: T,
  max_gap_num_4_il: T,
  sta_part_func_mats: &StaPartFuncMats<T>,
  produces_struct_profs: bool,
  global_part_func: PartFunc,
  trains_score_params: bool,
  expected_feature_count_sets: &mut FeatureCountSets,
  pos_quadruple_mat: &PosQuadrupleMat<T>,
  bp_score_param_set_pair: &BpScoreParamSetPair<T>,
  produces_align_probs: bool,
) -> StaProbMats<T>
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
  let mut sta_outside_part_func_4d_mat_4_bpas = PartFunc4dMat::<T>::default();
  let mut sta_prob_mats = StaProbMats::<T>::new(&seq_len_pair);
  let leftmost_pos_pair = (T::zero(), T::zero());
  let rightmost_pos_pair = (seq_len_pair.0 - T::one(), seq_len_pair.1 - T::one());
  for substr_len_1 in range_inclusive(
    T::from_usize(CONSPROB_MIN_HAIRPIN_LOOP_SPAN).unwrap(),
    max_bp_span_pair.0,
  )
  .rev()
  {
    for substr_len_2 in range_inclusive(
      T::from_usize(CONSPROB_MIN_HAIRPIN_LOOP_LEN).unwrap(),
      max_bp_span_pair.1,
    )
    .rev()
    {
      for &(i, j, k, l) in pos_quadruple_mat {
        if j - i + T::one() != substr_len_1 || l - k + T::one() != substr_len_2 {continue;}
        let pos_quadruple = (i, j, k, l);
        match sta_part_func_mats
          .part_func_4d_mat_4_bpas
          .get(&pos_quadruple)
        {
          Some(&part_func_4_bpa) => {
            let (long_i, long_j, long_k, long_l) = (i.to_usize().unwrap(), j.to_usize().unwrap(), k.to_usize().unwrap(), l.to_usize().unwrap());
            let base_pair = (seq_pair.0[long_i], seq_pair.0[long_j]);
            let base_pair_2 = (seq_pair.1[long_k], seq_pair.1[long_l]);
            let mismatch_pair = (seq_pair.0[long_j + 1], seq_pair.0[long_i - 1]);
            let mismatch_pair_2 = (seq_pair.1[long_l + 1], seq_pair.1[long_k - 1]);
            let prob_coeff = part_func_4_bpa - global_part_func;
            let mut sum = NEG_INFINITY;
            let mut forward_term = sum;
            let mut forward_term_4_align = sum;
            let mut forward_term_4_insert = sum;
            let mut forward_term_4_insert_2 = sum;
            let mut backward_term = sum;
            match sta_part_func_mats
              .forward_part_func_set_mat_4_external_loop
              .get(&(i - T::one(), k - T::one()))
            {
              Some(part_funcs) => {
                let is_begin = (i - T::one(), k - T::one()) == leftmost_pos_pair;
                forward_term_4_align = part_funcs.part_func_4_align + if is_begin {0.} else {feature_score_sets.match_2_match_count};
                logsumexp(&mut forward_term, forward_term_4_align);
                forward_term_4_insert = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count;
                logsumexp(&mut forward_term, forward_term_4_insert);
                forward_term_4_insert_2 = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count;
                logsumexp(&mut forward_term, forward_term_4_insert_2);
              }
              None => {}
            }
            match sta_part_func_mats
              .backward_part_func_set_mat_4_external_loop
              .get(&(j + T::one(), l + T::one()))
            {
              Some(part_funcs) => {
                let is_end = (j + T::one(), l + T::one()) == rightmost_pos_pair;
                let term = part_funcs.part_func_4_align + if is_end {0.} else {feature_score_sets.match_2_match_count};
                logsumexp(&mut backward_term, term);
                let term = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count;
                logsumexp(&mut backward_term, term);
                let term = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count;
                logsumexp(&mut backward_term, term);
              }
              None => {}
            }
            let coefficient = sta_part_func_mats.part_func_4d_mat_4_bpas_accessible_on_els
              [&pos_quadruple]
              - part_func_4_bpa;
            if trains_score_params {
              let prob = prob_coeff + coefficient + forward_term_4_align + backward_term;
              logsumexp(&mut expected_feature_count_sets.match_2_match_count, prob);
              let prob = prob_coeff + coefficient + forward_term_4_insert + backward_term;
              logsumexp(&mut expected_feature_count_sets.match_2_insert_count, prob);
              let prob = prob_coeff + coefficient + forward_term_4_insert_2 + backward_term;
              logsumexp(&mut expected_feature_count_sets.match_2_insert_count, prob);
            }
            let part_func_4_el = forward_term + backward_term;
            if part_func_4_el > NEG_INFINITY {
              sum = coefficient + part_func_4_el;
              let bpap_4_el = prob_coeff + sum;
              if trains_score_params {
                // Count external loop accessible basepairings.
                logsumexp(&mut expected_feature_count_sets
                  .external_loop_accessible_basepairing_count, (2. as Prob).ln() + bpap_4_el);
                // Count helix ends.
                logsumexp(&mut expected_feature_count_sets.helix_end_count_mat[base_pair.1][base_pair.0], bpap_4_el);
                logsumexp(&mut expected_feature_count_sets.helix_end_count_mat[base_pair_2.1][base_pair_2.0], bpap_4_el);
                // Count external loop terminal mismatches.
                if j < seq_len_pair.0 - T::from_usize(2).unwrap() {
                  logsumexp(&mut expected_feature_count_sets.left_dangle_count_mat
                    [base_pair.1][base_pair.0][seq_pair.0[long_j + 1]], bpap_4_el);
                }
                if i > T::one() {
                  logsumexp(&mut expected_feature_count_sets.right_dangle_count_mat
                    [base_pair.1][base_pair.0][seq_pair.0[long_i - 1]], bpap_4_el);
                }
                if l < seq_len_pair.1 - T::from_usize(2).unwrap() {
                  logsumexp(&mut expected_feature_count_sets.left_dangle_count_mat
                    [base_pair_2.1][base_pair_2.0][seq_pair.1[long_l + 1]], bpap_4_el);
                }
                if k > T::one() {
                  logsumexp(&mut expected_feature_count_sets.right_dangle_count_mat
                    [base_pair_2.1][base_pair_2.0][seq_pair.1[long_k - 1]], bpap_4_el);
                }
              }
            }
            for &(m, n, o, p) in pos_quadruple_mat {
              if !(m < i && j < n) || !(o < k && l < p) {continue;}
              let (long_m, long_n, long_o, long_p ) = (m.to_usize().unwrap(), n.to_usize().unwrap(), o.to_usize().unwrap(), p.to_usize().unwrap());
              let loop_len_pair = (long_i - long_m - 1, long_n - long_j - 1);
              let loop_len_pair_2 = (long_k - long_o - 1, long_p - long_l - 1);
              if loop_len_pair.0 + loop_len_pair.1 > CONSPROB_MAX_TWOLOOP_LEN {
                continue;
              }
              if loop_len_pair_2.0 + loop_len_pair_2.1 > CONSPROB_MAX_TWOLOOP_LEN {
                continue;
              }
              let base_pair_3 = (seq_pair.0[long_m], seq_pair.0[long_n]);
              let base_pair_4 = (seq_pair.1[long_o], seq_pair.1[long_p]);
              let is_stack = loop_len_pair.0 == 0 && loop_len_pair.1 == 0;
              let is_bulge_loop = (loop_len_pair.0 == 0 || loop_len_pair.1 == 0) && !is_stack;
              let mismatch_pair_3 = (seq_pair.0[long_m + 1], seq_pair.0[long_n - 1]);
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
                  let mut forward_term_4_align = forward_term;
                  let mut forward_term_4_insert = forward_term;
                  let mut forward_term_4_insert_2 = forward_term;
                  let mut backward_term = forward_term;
                  match forward_tmp_part_func_set_mat.get(&(i - T::one(), k - T::one())) {
                    Some(part_func_sets) => {
                      let ref part_funcs = part_func_sets.part_funcs_on_sa;
                      forward_term_4_align = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
                      logsumexp(&mut forward_term, forward_term_4_align);
                      forward_term_4_insert = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count;
                      logsumexp(&mut forward_term, forward_term_4_insert);
                      forward_term_4_insert_2 = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count;
                      logsumexp(&mut forward_term, forward_term_4_insert_2);
                    }
                    None => {}
                  }
                  match backward_tmp_part_func_set_mat.get(&(j + T::one(), l + T::one())) {
                    Some(part_func_sets) => {
                      let ref part_funcs = part_func_sets.part_funcs_on_sa;
                      let term = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
                      logsumexp(&mut backward_term, term);
                      let term = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count;
                      logsumexp(&mut backward_term, term);
                      let term = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count;
                      logsumexp(&mut backward_term, term);
                    }
                    None => {}
                  }
                  let basepair_align_score = feature_score_sets.basepair_align_count_mat[base_pair_3.0][base_pair_3.1][base_pair_4.0][base_pair_4.1] + feature_score_sets.align_count_mat[base_pair_3.0][base_pair_4.0] + feature_score_sets.align_count_mat[base_pair_3.1][base_pair_4.1];
                  let twoloop_score = bp_score_param_set_pair.0.twoloop_scores[&(m, n, i, j)];
                  let twoloop_score_2 = bp_score_param_set_pair.1.twoloop_scores[&(o, p, k, l)];
                  let coefficient = basepair_align_score + twoloop_score + twoloop_score_2 + part_func;
                  if trains_score_params {
                    let prob = prob_coeff + coefficient + forward_term_4_align + backward_term;
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, prob);
                    let prob = prob_coeff + coefficient + forward_term_4_insert + backward_term;
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, prob);
                    let prob = prob_coeff + coefficient + forward_term_4_insert_2 + backward_term;
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, prob);
                  }
                  let part_func_4_2l = forward_term + backward_term;
                  if part_func_4_2l > NEG_INFINITY {
                    let part_func_4_2l = coefficient + part_func_4_2l;
                    logsumexp(&mut sum, part_func_4_2l);
                    let bpap_4_2l = prob_coeff + part_func_4_2l;
                    if produces_struct_profs {
                      for q in long_m + 1 .. long_i {
                        logsumexp(&mut sta_prob_mats.upp_mat_pair_4_2l.0[q], bpap_4_2l);
                      }
                      for q in long_j + 1 .. long_n {
                        logsumexp(&mut sta_prob_mats.upp_mat_pair_4_2l.0[q], bpap_4_2l);
                      }
                      for q in long_o + 1 .. long_k {
                        logsumexp(&mut sta_prob_mats.upp_mat_pair_4_2l.1[q], bpap_4_2l);
                      }
                      for q in long_l + 1 .. long_p {
                        logsumexp(&mut sta_prob_mats.upp_mat_pair_4_2l.1[q], bpap_4_2l);
                      }
                    }
                    if trains_score_params {
                      if is_stack {
                        // Count a stack.
                        let dict_min_stack = get_dict_min_stack(&base_pair_3, &base_pair);
                        logsumexp(&mut expected_feature_count_sets.stack_count_mat[dict_min_stack.0.0]
                          [dict_min_stack.0.1][dict_min_stack.1.0][dict_min_stack.1.1], bpap_4_2l);
                      } else {
                        if is_bulge_loop {
                          // Count a bulge loop length.
                          let bulge_loop_len = if loop_len_pair.0 == 0 {
                            loop_len_pair.1
                          } else {
                            loop_len_pair.0
                          };
                          logsumexp(&mut expected_feature_count_sets.bulge_loop_length_counts
                            [bulge_loop_len - 1], bpap_4_2l);
                          // Count a 0x1 bulge loop.
                          if bulge_loop_len == 1 {
                            let mismatch = if loop_len_pair.0 == 0 {mismatch_pair_3.1} else {mismatch_pair_3.0};
                            logsumexp(&mut expected_feature_count_sets.bulge_loop_0x1_length_counts
                              [mismatch], bpap_4_2l);
                          }
                        } else {
                          // Count an interior loop length.
                          logsumexp(&mut expected_feature_count_sets.interior_loop_length_counts
                            [loop_len_pair.0 + loop_len_pair.1 - 2], bpap_4_2l);
                          let diff = get_diff(loop_len_pair.0, loop_len_pair.1);
                          if diff == 0 {
                            logsumexp(&mut expected_feature_count_sets.interior_loop_length_counts_symm
                              [loop_len_pair.0 - 1], bpap_4_2l);
                          } else {
                            logsumexp(&mut expected_feature_count_sets.interior_loop_length_counts_asymm
                              [diff - 1], bpap_4_2l);
                          }
                          // Count a 1x1 interior loop.
                          if loop_len_pair.0 == 1 && loop_len_pair.1 == 1 {
                            let dict_min_mismatch_pair_3 = get_dict_min_mismatch_pair(&mismatch_pair_3);
                            logsumexp(&mut expected_feature_count_sets.interior_loop_1x1_length_count_mat
                              [dict_min_mismatch_pair_3.0][dict_min_mismatch_pair_3.1], bpap_4_2l);
                          }
                          // Count an explicit interior loop length pair.
                          if loop_len_pair.0 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT && loop_len_pair.1 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT {
                            let dict_min_loop_len_pair =  get_dict_min_loop_len_pair(&loop_len_pair);
                            logsumexp(&mut expected_feature_count_sets.interior_loop_length_count_mat_explicit
                              [dict_min_loop_len_pair.0 - 1][dict_min_loop_len_pair.1 - 1], bpap_4_2l);
                          }
                        }
                        // Count helix ends.
                        logsumexp(&mut expected_feature_count_sets.helix_end_count_mat[base_pair.1][base_pair.0], bpap_4_2l);
                        logsumexp(&mut expected_feature_count_sets.helix_end_count_mat[base_pair_3.0][base_pair_3.1], bpap_4_2l);
                        // Count 2-loop terminal mismatches.
                        logsumexp(&mut expected_feature_count_sets.terminal_mismatch_count_mat
                          [base_pair_3.0][base_pair_3.1][mismatch_pair_3.0]
                          [mismatch_pair_3.1], bpap_4_2l);
                        logsumexp(&mut expected_feature_count_sets.terminal_mismatch_count_mat
                          [base_pair.1][base_pair.0][mismatch_pair.0][mismatch_pair.1],
                          bpap_4_2l);
                      }
                      if is_stack_2 {
                        // Count a stack.
                        let dict_min_stack_2 = get_dict_min_stack(&base_pair_4, &base_pair_2);
                        logsumexp(&mut expected_feature_count_sets.stack_count_mat[dict_min_stack_2.0.0]
                          [dict_min_stack_2.0.1][dict_min_stack_2.1.0][dict_min_stack_2.1.1], bpap_4_2l);
                      } else {
                        if is_bulge_loop_2 {
                          // Count a bulge loop length.
                          let bulge_loop_len_2 = if loop_len_pair_2.0 == 0 {
                            loop_len_pair_2.1
                          } else {
                            loop_len_pair_2.0
                          };
                          logsumexp(&mut expected_feature_count_sets.bulge_loop_length_counts
                            [bulge_loop_len_2 - 1], bpap_4_2l);
                          // Count a 0x1 bulge loop.
                          if bulge_loop_len_2 == 1 {
                            let mismatch_2 = if loop_len_pair_2.0 == 0 {mismatch_pair_4.1} else {mismatch_pair_4.0};
                            logsumexp(&mut expected_feature_count_sets.bulge_loop_0x1_length_counts
                              [mismatch_2], bpap_4_2l);
                          }
                        } else {
                          // Count an interior loop length.
                          logsumexp(&mut expected_feature_count_sets.interior_loop_length_counts
                            [loop_len_pair_2.0 + loop_len_pair_2.1 - 2], bpap_4_2l);
                          let diff_2 = get_diff(loop_len_pair_2.0, loop_len_pair_2.1);
                          if diff_2 == 0 {
                            logsumexp(&mut expected_feature_count_sets.interior_loop_length_counts_symm
                              [loop_len_pair_2.0 - 1], bpap_4_2l);
                          } else {
                            logsumexp(&mut expected_feature_count_sets.interior_loop_length_counts_asymm
                              [diff_2 - 1], bpap_4_2l);
                          }
                          // Count a 1x1 interior loop.
                          if loop_len_pair_2.0 == 1 && loop_len_pair_2.1 == 1 {
                            let dict_min_mismatch_pair_4 = get_dict_min_mismatch_pair(&mismatch_pair_4);
                            logsumexp(&mut expected_feature_count_sets.interior_loop_1x1_length_count_mat
                              [dict_min_mismatch_pair_4.0][dict_min_mismatch_pair_4.1], bpap_4_2l);
                          }
                          // Count an explicit interior loop length pair.
                          if loop_len_pair_2.0 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT && loop_len_pair_2.1 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT {
                            let dict_min_loop_len_pair_2 =  get_dict_min_loop_len_pair(&loop_len_pair_2);
                            logsumexp(&mut expected_feature_count_sets.interior_loop_length_count_mat_explicit
                              [dict_min_loop_len_pair_2.0 - 1][dict_min_loop_len_pair_2.1 - 1], bpap_4_2l);
                          }
                        }
                        // Count helix ends.
                        logsumexp(&mut expected_feature_count_sets.helix_end_count_mat[base_pair_2.1][base_pair_2.0], bpap_4_2l);
                        logsumexp(&mut expected_feature_count_sets.helix_end_count_mat[base_pair_4.0][base_pair_4.1], bpap_4_2l);
                        // Count 2-loop terminal mismatches.
                        logsumexp(&mut expected_feature_count_sets.terminal_mismatch_count_mat
                          [base_pair_4.0][base_pair_4.1][mismatch_pair_4.0]
                          [mismatch_pair_4.1], bpap_4_2l);
                        logsumexp(&mut expected_feature_count_sets.terminal_mismatch_count_mat
                          [base_pair_2.1][base_pair_2.0][mismatch_pair_2.0]
                          [mismatch_pair_2.1], bpap_4_2l);
                      }
                    }
                  }
                }
                None => {}
              }
            }
            let part_func_ratio = sta_part_func_mats.part_func_4d_mat_4_bpas_accessible_on_mls
              [&pos_quadruple]
              - part_func_4_bpa;
            for &(m, n, o, p) in pos_quadruple_mat {
              if !(m < i && j < n) || !(o < k && l < p) {continue;}
              let long_m = m.to_usize().unwrap();
              let long_n = n.to_usize().unwrap();
              let base_pair_3 = (seq_pair.0[long_m], seq_pair.0[long_n]);
              let mismatch_pair_3 = (seq_pair.0[long_m + 1], seq_pair.0[long_n - 1]);
              let base_pair = (seq_pair.0[long_m], seq_pair.0[long_n]);
              let long_o = o.to_usize().unwrap();
              let long_p = p.to_usize().unwrap();
              let base_pair_4 = (seq_pair.1[long_o], seq_pair.1[long_p]);
              let pos_quadruple_2 = (m, n, o, p);
              match sta_outside_part_func_4d_mat_4_bpas.get(&pos_quadruple_2) {
                Some(&part_func_4_bpa_2) => {
                  let mismatch_pair_4 = (seq_pair.1[long_o + 1], seq_pair.1[long_p - 1]);
                  let base_pair_2 = (seq_pair.1[long_o], seq_pair.1[long_p]);
                  let ref forward_tmp_part_func_set_mat = sta_part_func_mats
                    .forward_tmp_part_func_set_mats_with_pos_pairs[&(m, o)];
                  let ref backward_tmp_part_func_set_mat = sta_part_func_mats
                    .backward_tmp_part_func_set_mats_with_pos_pairs[&(n, p)];
                  let basepair_align_score = feature_score_sets.basepair_align_count_mat[base_pair_3.0][base_pair_3.1][base_pair_4.0][base_pair_4.1] + feature_score_sets.align_count_mat[base_pair_3.0][base_pair_4.0] + feature_score_sets.align_count_mat[base_pair_3.1][base_pair_4.1];
                  let multi_loop_closing_basepairing_score = bp_score_param_set_pair.0.multi_loop_closing_bp_scores[&(m, n)];
                  let multi_loop_closing_basepairing_score_2 = bp_score_param_set_pair.1.multi_loop_closing_bp_scores[&(o, p)];
                  let mut forward_term = NEG_INFINITY;
                  let mut forward_term_4_align = forward_term;
                  let mut forward_term_4_insert = forward_term;
                  let mut forward_term_4_insert_2 = forward_term;
                  let mut forward_term_2 = forward_term;
                  let mut forward_term_2_4_align = forward_term;
                  let mut forward_term_2_4_insert = forward_term;
                  let mut forward_term_2_4_insert_2 = forward_term;
                  let mut backward_term = forward_term;
                  let mut backward_term_2 = forward_term;
                  match forward_tmp_part_func_set_mat.get(&(i - T::one(), k - T::one())) {
                    Some(part_func_sets) => {
                      let ref part_funcs = part_func_sets.part_funcs_4_bpas_on_mls;
                      forward_term_4_align = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
                      logsumexp(&mut forward_term, forward_term_4_align);
                      forward_term_4_insert = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count;
                      logsumexp(&mut forward_term, forward_term_4_insert);
                      forward_term_4_insert_2 = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count;
                      logsumexp(&mut forward_term, forward_term_4_insert_2);
                      let ref part_funcs = part_func_sets.part_funcs_on_sa_4_ml;
                      forward_term_2_4_align = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
                      logsumexp(&mut forward_term_2, forward_term_2_4_align);
                      forward_term_2_4_insert = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count;
                      logsumexp(&mut forward_term_2, forward_term_2_4_insert);
                      forward_term_2_4_insert_2 = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count;
                      logsumexp(&mut forward_term_2, forward_term_2_4_insert_2);
                    }
                    None => {}
                  }
                  match backward_tmp_part_func_set_mat.get(&(j + T::one(), l + T::one())) {
                    Some(part_func_sets) => {
                      let ref part_funcs = part_func_sets.part_funcs_on_mls;
                      let term = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
                      logsumexp(&mut backward_term, term);
                      let term = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count;
                      logsumexp(&mut backward_term, term);
                      let term = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count;
                      logsumexp(&mut backward_term, term);
                      let ref part_funcs = part_func_sets.part_funcs_4_bpas_on_mls;
                      let term = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
                      logsumexp(&mut backward_term_2, term);
                      let term = part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count;
                      logsumexp(&mut backward_term_2, term);
                      let term = part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count;
                      logsumexp(&mut backward_term_2, term);
                    }
                    None => {}
                  }
                  let coefficient = part_func_ratio
                    + basepair_align_score
                    + multi_loop_closing_basepairing_score
                    + multi_loop_closing_basepairing_score_2
                    + part_func_4_bpa_2;
                  if trains_score_params {
                    let prob = prob_coeff + coefficient + forward_term_4_align + backward_term;
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, prob);
                    let prob = prob_coeff + coefficient + forward_term_4_insert + backward_term;
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, prob);
                    let prob = prob_coeff + coefficient + forward_term_4_insert_2 + backward_term;
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, prob);
                    let prob = prob_coeff + coefficient + forward_term_2_4_align + backward_term_2;
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, prob);
                    let prob = prob_coeff + coefficient + forward_term_2_4_insert + backward_term_2;
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, prob);
                    let prob = prob_coeff + coefficient + forward_term_2_4_insert_2 + backward_term_2;
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, prob);
                  }
                  let mut part_func_4_ml = forward_term + backward_term;
                  logsumexp(
                    &mut part_func_4_ml,
                    forward_term_2 + backward_term_2,
                  );
                  if part_func_4_ml > NEG_INFINITY {
                    let part_func_4_ml = coefficient + part_func_4_ml;
                    logsumexp(&mut sum, part_func_4_ml);
                    let bpap_4_ml = prob_coeff + part_func_4_ml;
                    if trains_score_params {
                      // Count multi-loop terminal mismatches.
                      logsumexp(&mut expected_feature_count_sets.left_dangle_count_mat
                        [base_pair_3.0][base_pair_3.1][mismatch_pair_3.0],
                        bpap_4_ml);
                      logsumexp(&mut expected_feature_count_sets.right_dangle_count_mat
                        [base_pair_3.0][base_pair_3.1][mismatch_pair_3.1],
                        bpap_4_ml);
                      logsumexp(&mut expected_feature_count_sets.left_dangle_count_mat[base_pair_4.0]
                        [base_pair_4.1][mismatch_pair_4.0], bpap_4_ml);
                      logsumexp(&mut expected_feature_count_sets.left_dangle_count_mat[base_pair_4.0]
                        [base_pair_4.1][mismatch_pair_4.1], bpap_4_ml);
                      logsumexp(&mut expected_feature_count_sets.left_dangle_count_mat
                        [base_pair.1][base_pair.0][mismatch_pair.0],
                        bpap_4_ml);
                      logsumexp(&mut expected_feature_count_sets.right_dangle_count_mat
                        [base_pair.1][base_pair.0][mismatch_pair.1],
                        bpap_4_ml);
                      logsumexp(&mut expected_feature_count_sets.left_dangle_count_mat[base_pair_2.1]
                        [base_pair_2.0][mismatch_pair_2.0], bpap_4_ml);
                      logsumexp(&mut expected_feature_count_sets.right_dangle_count_mat[base_pair_2.1]
                        [base_pair_2.0][mismatch_pair_2.1], bpap_4_ml);
                      // Count helix ends.
                      logsumexp(&mut expected_feature_count_sets.helix_end_count_mat
                        [base_pair_3.0][base_pair_3.1], bpap_4_ml);
                      logsumexp(&mut expected_feature_count_sets.helix_end_count_mat[base_pair_4.0]
                        [base_pair_4.1], bpap_4_ml);
                      logsumexp(&mut expected_feature_count_sets.helix_end_count_mat
                        [base_pair.1][base_pair.0],
                        bpap_4_ml);
                      logsumexp(&mut expected_feature_count_sets.helix_end_count_mat[base_pair_2.1]
                        [base_pair_2.0], bpap_4_ml);
                      // Count multi-loop closings.
                      logsumexp(&mut expected_feature_count_sets.multi_loop_base_count, (2. as Prob).ln() + bpap_4_ml);
                      // Count multi-loop closing basepairings.
                      logsumexp(&mut expected_feature_count_sets.multi_loop_basepairing_count, (2. as Prob).ln() + bpap_4_ml);
                      // Count multi-loop accessible basepairings.
                      logsumexp(&mut expected_feature_count_sets
                        .multi_loop_basepairing_count, (2. as Prob).ln() + bpap_4_ml);
                    }
                  }
                }
                None => {}
              }
            }
            if sum > NEG_INFINITY {
              sta_outside_part_func_4d_mat_4_bpas.insert(pos_quadruple, sum);
              let bpap = prob_coeff + sum;
              if produces_align_probs {
                sta_prob_mats.basepair_align_prob_mat.insert(pos_quadruple, bpap);
              }
              if trains_score_params {
                // Count base pairs.
                let dict_min_base_pair = get_dict_min_base_pair(&base_pair);
                let dict_min_base_pair_2 = get_dict_min_base_pair(&base_pair_2);
                logsumexp(&mut expected_feature_count_sets.base_pair_count_mat[dict_min_base_pair.0][dict_min_base_pair.1], bpap);
                logsumexp(&mut expected_feature_count_sets.base_pair_count_mat[dict_min_base_pair_2.0][dict_min_base_pair_2.1], bpap);
                // Count a basepair alignment.
                let dict_min_basepair_align = get_dict_min_basepair_align(&base_pair, &base_pair_2);
                logsumexp(&mut expected_feature_count_sets.basepair_align_count_mat[dict_min_basepair_align.0.0][dict_min_basepair_align.0.1]
                  [dict_min_basepair_align.1.0][dict_min_basepair_align.1.1], bpap);
                // Count alignments.
                let dict_min_align = get_dict_min_align(&(base_pair.0, base_pair_2.0));
                logsumexp(&mut expected_feature_count_sets.align_count_mat[dict_min_align.0][dict_min_align.1], bpap);
                let dict_min_align = get_dict_min_align(&(base_pair.1, base_pair_2.1));
                logsumexp(&mut expected_feature_count_sets.align_count_mat[dict_min_align.0][dict_min_align.1], bpap);
              }
              match sta_prob_mats.bpp_mat_pair.0.get_mut(&(i, j)) {
                Some(bpp) => {
                  logsumexp(bpp, bpap);
                }
                None => {
                  sta_prob_mats.bpp_mat_pair.0.insert((i, j), bpap);
                }
              }
              match sta_prob_mats.bpp_mat_pair.1.get_mut(&(k, l)) {
                Some(bpp) => {
                  logsumexp(bpp, bpap);
                }
                None => {
                  sta_prob_mats.bpp_mat_pair.1.insert((k, l), bpap);
                }
              }
              logsumexp(&mut sta_prob_mats.bpp_mat_pair_2.0[long_i], bpap);
              logsumexp(&mut sta_prob_mats.bpp_mat_pair_2.0[long_j], bpap);
              logsumexp(&mut sta_prob_mats.bpp_mat_pair_2.1[long_k], bpap);
              logsumexp(&mut sta_prob_mats.bpp_mat_pair_2.1[long_l], bpap);
              if trains_score_params && substr_len_1.to_usize().unwrap() - 2 <= CONSPROB_MAX_HAIRPIN_LOOP_LEN && substr_len_2.to_usize().unwrap() - 2 <= CONSPROB_MAX_HAIRPIN_LOOP_LEN {
                let ref part_funcs = sta_part_func_mats.forward_tmp_part_func_set_mats_with_pos_pairs[&(i, k)][&(j - T::one(), l - T::one())].part_funcs_on_sa;
                let mut part_func_on_sa = NEG_INFINITY;
                logsumexp(&mut part_func_on_sa, part_funcs.part_func_4_align + feature_score_sets.match_2_match_count);
                logsumexp(&mut part_func_on_sa, part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count);
                logsumexp(&mut part_func_on_sa, part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count);
                let hairpin_loop_score = bp_score_param_set_pair.0.hairpin_loop_scores[&(i, j)];
                let hairpin_loop_score_2 = bp_score_param_set_pair.1.hairpin_loop_scores[&(k, l)];
                let bpap_4_hl = sum - global_part_func + part_func_on_sa + hairpin_loop_score + hairpin_loop_score_2;
                logsumexp(&mut expected_feature_count_sets.hairpin_loop_length_counts[long_j - long_i - 1 - CONSPROB_MIN_HAIRPIN_LOOP_LEN], bpap_4_hl);
                logsumexp(&mut expected_feature_count_sets.hairpin_loop_length_counts[long_l - long_k - 1 - CONSPROB_MIN_HAIRPIN_LOOP_LEN], bpap_4_hl);
                let mismatch_pair = (seq_pair.0[long_i + 1], seq_pair.0[long_j - 1]);
                logsumexp(&mut expected_feature_count_sets.terminal_mismatch_count_mat
                  [base_pair.0][base_pair.1][mismatch_pair.0]
                  [mismatch_pair.1], bpap_4_hl);
                let mismatch_pair_2 = (seq_pair.1[long_k + 1], seq_pair.1[long_l - 1]);
                logsumexp(&mut expected_feature_count_sets.terminal_mismatch_count_mat
                  [base_pair_2.0][base_pair_2.1][mismatch_pair_2.0]
                  [mismatch_pair_2.1], bpap_4_hl);
                // Count helix ends.
                logsumexp(&mut expected_feature_count_sets.helix_end_count_mat[base_pair.0][base_pair.1], bpap_4_hl);
                logsumexp(&mut expected_feature_count_sets.helix_end_count_mat[base_pair_2.0][base_pair_2.1], bpap_4_hl);
              }
            }
          }
          None => {}
        }
      }
    }
  }
  for bpp in sta_prob_mats.bpp_mat_pair.0.values_mut() {
    *bpp = expf(*bpp);
  }
  for bpp in sta_prob_mats.bpp_mat_pair.1.values_mut() {
    *bpp = expf(*bpp);
  }
  if produces_struct_profs || produces_align_probs || trains_score_params {
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
        let pos_pair_2 = (u + T::one(), v + T::one());
        let mut backward_term_4_align = NEG_INFINITY;
        let mut backward_term_4_insert = backward_term_4_align;
        let mut backward_term_4_insert_2 = backward_term_4_align;
        match sta_part_func_mats
          .backward_part_func_set_mat_4_external_loop.get(&pos_pair_2) {
          Some(part_funcs_2) => {
            let is_end = pos_pair_2 == rightmost_pos_pair;
            logsumexp(&mut backward_term_4_align, part_funcs_2.part_func_4_align + if is_end {0.} else {feature_score_sets.match_2_match_count});
            logsumexp(&mut backward_term_4_align, part_funcs_2.part_func_4_insert + feature_score_sets.match_2_insert_count);
            logsumexp(&mut backward_term_4_align, part_funcs_2.part_func_4_insert_2 + feature_score_sets.match_2_insert_count);
            logsumexp(&mut backward_term_4_insert, part_funcs_2.part_func_4_align + if is_end {0.} else {feature_score_sets.match_2_insert_count});
            logsumexp(&mut backward_term_4_insert, part_funcs_2.part_func_4_insert + feature_score_sets.insert_extend_count);
            logsumexp(&mut backward_term_4_insert, part_funcs_2.part_func_4_insert_2 + feature_score_sets.insert_switch_count);
            logsumexp(&mut backward_term_4_insert_2, part_funcs_2.part_func_4_align + if is_end {0.} else {feature_score_sets.match_2_insert_count});
            logsumexp(&mut backward_term_4_insert_2, part_funcs_2.part_func_4_insert + feature_score_sets.insert_switch_count);
            logsumexp(&mut backward_term_4_insert_2, part_funcs_2.part_func_4_insert_2 + feature_score_sets.insert_extend_count);
          }, None => {},
        }
        let dict_min_loop_align = get_dict_min_loop_align(&(base, base_2));
        if u > T::zero() && v > T::zero() {
          match sta_part_func_mats
            .forward_part_func_set_mat_4_external_loop
            .get(&pos_pair_4_loop_align)
          {
            Some(part_funcs) => {
              let is_begin = pos_pair_4_loop_align == leftmost_pos_pair;
              let loop_align_score = feature_score_sets.loop_align_count_mat[base][base_2] + feature_score_sets.align_count_mat[base][base_2];
              let mut loop_align_prob_4_el = NEG_INFINITY;
              let term = loop_align_score + part_funcs.part_func_4_align + if is_begin {0.} else {feature_score_sets.match_2_match_count} + backward_term_4_align - global_part_func + 2. * feature_score_sets.external_loop_accessible_baseunpairing_count;
              if trains_score_params {
                if !is_begin {
                  logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
                }
              }
              logsumexp(&mut loop_align_prob_4_el, term);
              let term = loop_align_score + part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + backward_term_4_align - global_part_func + 2. * feature_score_sets.external_loop_accessible_baseunpairing_count;
              if trains_score_params {
                logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
              }
              logsumexp(&mut loop_align_prob_4_el, term);
              let term = loop_align_score + part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + backward_term_4_align - global_part_func + 2. * feature_score_sets.external_loop_accessible_baseunpairing_count;
              if trains_score_params {
                logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
              }
              logsumexp(&mut loop_align_prob_4_el, term);
              if produces_struct_profs {
                logsumexp(&mut sta_prob_mats.upp_mat_pair_4_el.0[long_u], loop_align_prob_4_el);
                logsumexp(&mut sta_prob_mats.upp_mat_pair_4_el.1[long_v], loop_align_prob_4_el);
              }
              if produces_align_probs {
                match sta_prob_mats.loop_align_prob_mat.get_mut(&pos_pair) {
                  Some(loop_align_prob) => {
                    logsumexp(loop_align_prob, loop_align_prob_4_el);
                  }
                  None => {
                    sta_prob_mats.loop_align_prob_mat.insert(pos_pair, loop_align_prob_4_el);
                  }
                }
              }
              if trains_score_params {
                logsumexp(&mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_el);
                logsumexp(&mut expected_feature_count_sets.align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_el);
                logsumexp(&mut expected_feature_count_sets.external_loop_accessible_baseunpairing_count, (2. as Prob).ln() + loop_align_prob_4_el);
              }
            }, None => {},
          }
        }
        if u > T::zero() {
          let insert_score = feature_score_sets.insert_counts[base];
          match sta_part_func_mats
            .forward_part_func_set_mat_4_external_loop
            .get(&pos_pair_4_insert)
          {
            Some(part_funcs) => {
              let is_begin = pos_pair_4_insert == leftmost_pos_pair;
              let mut insert_prob = NEG_INFINITY;
              let term = insert_score + if is_begin {0.} else {feature_score_sets.match_2_insert_count} + part_funcs.part_func_4_align + backward_term_4_insert - global_part_func + feature_score_sets.external_loop_accessible_baseunpairing_count;
              if trains_score_params {
                if !is_begin {
                  logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                }
              }
              logsumexp(&mut insert_prob, term);
              let term = insert_score + feature_score_sets.insert_extend_count + part_funcs.part_func_4_insert + backward_term_4_insert - global_part_func + feature_score_sets.external_loop_accessible_baseunpairing_count;
              if trains_score_params {
                logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
              }
              logsumexp(&mut insert_prob, term);
              let term = insert_score + feature_score_sets.insert_switch_count + part_funcs.part_func_4_insert_2 + backward_term_4_insert - global_part_func + feature_score_sets.external_loop_accessible_baseunpairing_count;
              if trains_score_params {
                logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
              }
              logsumexp(&mut insert_prob, term);
              if produces_struct_profs {
                logsumexp(&mut sta_prob_mats.upp_mat_pair_4_el.0[long_u], insert_prob);
              }
              if trains_score_params {
                logsumexp(&mut expected_feature_count_sets.insert_counts[base], insert_prob);
                logsumexp(&mut expected_feature_count_sets.external_loop_accessible_baseunpairing_count, insert_prob);
              }
            },
            None => {},
          }
        }
        if v > T::zero() {
          let insert_score_2 = feature_score_sets.insert_counts[base_2];
          match sta_part_func_mats
            .forward_part_func_set_mat_4_external_loop
            .get(&pos_pair_4_insert_2)
          {
            Some(part_funcs) => {
              let is_begin = pos_pair_4_insert_2 == leftmost_pos_pair;
              let mut insert_prob = NEG_INFINITY;
              let term = insert_score_2 + if is_begin {0.} else {feature_score_sets.match_2_insert_count} + part_funcs.part_func_4_align + backward_term_4_insert_2 - global_part_func + feature_score_sets.external_loop_accessible_baseunpairing_count;
              if trains_score_params {
                if !is_begin {
                  logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                }
              }
              logsumexp(&mut insert_prob, term);
              let term = insert_score_2 + feature_score_sets.insert_switch_count + part_funcs.part_func_4_insert + backward_term_4_insert_2 - global_part_func + feature_score_sets.external_loop_accessible_baseunpairing_count;
              if trains_score_params {
                logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
              }
              logsumexp(&mut insert_prob, term);
              let term = insert_score_2 + feature_score_sets.insert_extend_count + part_funcs.part_func_4_insert_2 + backward_term_4_insert_2 - global_part_func + feature_score_sets.external_loop_accessible_baseunpairing_count;
              if trains_score_params {
                logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
              }
              logsumexp(&mut insert_prob, term);
              if produces_struct_profs {
                logsumexp(&mut sta_prob_mats.upp_mat_pair_4_el.1[long_v], insert_prob);
              }
              if trains_score_params {
                logsumexp(&mut expected_feature_count_sets.insert_counts[base_2], insert_prob);
                logsumexp(&mut expected_feature_count_sets.external_loop_accessible_baseunpairing_count, insert_prob);
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
        let loop_align_score = feature_score_sets.loop_align_count_mat[base][base_2] + feature_score_sets.align_count_mat[base][base_2];
        let insert_score = feature_score_sets.insert_counts[base];
        let insert_score_2 = feature_score_sets.insert_counts[base_2];
        for &(i, j, k, l) in pos_quadruple_mat {
          if !(i < u && u < j) || !(k < v && v < l) {
            continue;
          }
          let pos_quadruple = (i, j, k, l);
          match sta_outside_part_func_4d_mat_4_bpas.get(&pos_quadruple) {
            Some(&part_func_4_bpa) => {
              let (long_i, long_j, long_k, long_l) = (i.to_usize().unwrap(), j.to_usize().unwrap(), k.to_usize().unwrap(), l.to_usize().unwrap());
              let base_pair = (seq_pair.0[long_i], seq_pair.0[long_j]);
              let base_pair_2 = (seq_pair.1[long_k], seq_pair.1[long_l]);
              let hairpin_loop_score = if j - i - T::one() <= T::from_usize(CONSPROB_MAX_HAIRPIN_LOOP_LEN).unwrap() {bp_score_param_set_pair.0.hairpin_loop_scores[&(i, j)]} else {NEG_INFINITY};
              let hairpin_loop_score_2 = if l - k - T::one() <= T::from_usize(CONSPROB_MAX_HAIRPIN_LOOP_LEN).unwrap() {bp_score_param_set_pair.1.hairpin_loop_scores[&(k, l)]} else {NEG_INFINITY};
              let multi_loop_closing_basepairing_score = bp_score_param_set_pair.0.multi_loop_closing_bp_scores[&(i, j)];
              let multi_loop_closing_basepairing_score_2 = bp_score_param_set_pair.1.multi_loop_closing_bp_scores[&(k, l)];
              let basepair_align_score = feature_score_sets.basepair_align_count_mat[base_pair.0][base_pair.1][base_pair_2.0][base_pair_2.1] + feature_score_sets.align_count_mat[base_pair.0][base_pair_2.0] + feature_score_sets.align_count_mat[base_pair.1][base_pair_2.1];
              let prob_coeff = part_func_4_bpa - global_part_func + basepair_align_score;
              let ref forward_tmp_part_func_set_mat =
                sta_part_func_mats.forward_tmp_part_func_set_mats_with_pos_pairs[&(i, k)];
              let ref backward_tmp_part_func_set_mat =
                sta_part_func_mats.backward_tmp_part_func_set_mats_with_pos_pairs[&(j, l)];
              let ref forward_tmp_part_func_set_mat_4_2loop =
                sta_part_func_mats.forward_2loop_tmp_part_func_set_mats_with_pos_quadruples[&pos_quadruple];
              let ref backward_tmp_part_func_set_mat_4_2loop =
                sta_part_func_mats.backward_2loop_tmp_part_func_set_mats_with_pos_quadruples[&pos_quadruple];
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
              match backward_tmp_part_func_set_mat.get(&pos_pair_2) {
                Some(part_func_sets) => {
                  let ref part_funcs = part_func_sets.part_funcs_on_sa;
                  backward_term_4_align_on_sa = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
                  logsumexp(&mut backward_term_4_align_on_sa, part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count);
                  logsumexp(&mut backward_term_4_align_on_sa, part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count);
                  backward_term_4_insert_on_sa = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut backward_term_4_insert_on_sa, part_funcs.part_func_4_insert + feature_score_sets.insert_extend_count);
                  logsumexp(&mut backward_term_4_insert_on_sa, part_funcs.part_func_4_insert_2 + feature_score_sets.insert_switch_count);
                  backward_term_4_insert_on_sa_2 = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut backward_term_4_insert_on_sa_2, part_funcs.part_func_4_insert + feature_score_sets.insert_switch_count);
                  logsumexp(&mut backward_term_4_insert_on_sa_2, part_funcs.part_func_4_insert_2 + feature_score_sets.insert_extend_count);
                  let ref part_funcs = part_func_sets.part_funcs_4_ml;
                  backward_term_4_align_4_ml = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
                  logsumexp(&mut backward_term_4_align_4_ml, part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count);
                  logsumexp(&mut backward_term_4_align_4_ml, part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count);
                  backward_term_4_insert_4_ml = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut backward_term_4_insert_4_ml, part_funcs.part_func_4_insert + feature_score_sets.insert_extend_count);
                  logsumexp(&mut backward_term_4_insert_4_ml, part_funcs.part_func_4_insert_2 + feature_score_sets.insert_switch_count);
                  backward_term_4_insert_4_ml_2 = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut backward_term_4_insert_4_ml_2, part_funcs.part_func_4_insert + feature_score_sets.insert_switch_count);
                  logsumexp(&mut backward_term_4_insert_4_ml_2, part_funcs.part_func_4_insert_2 + feature_score_sets.insert_extend_count);
                  let ref part_funcs = part_func_sets.part_funcs_4_bpas_on_mls;
                  backward_term_4_align_4_bpas_on_mls = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
                  logsumexp(&mut backward_term_4_align_4_bpas_on_mls, part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count);
                  logsumexp(&mut backward_term_4_align_4_bpas_on_mls, part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count);
                  backward_term_4_insert_4_bpas_on_mls = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut backward_term_4_insert_4_bpas_on_mls, part_funcs.part_func_4_insert + feature_score_sets.insert_extend_count);
                  logsumexp(&mut backward_term_4_insert_4_bpas_on_mls, part_funcs.part_func_4_insert_2 + feature_score_sets.insert_switch_count);
                  backward_term_4_insert_4_bpas_on_mls_2 = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut backward_term_4_insert_4_bpas_on_mls_2, part_funcs.part_func_4_insert + feature_score_sets.insert_switch_count);
                  logsumexp(&mut backward_term_4_insert_4_bpas_on_mls_2, part_funcs.part_func_4_insert_2 + feature_score_sets.insert_extend_count);
                  let ref part_funcs = part_func_sets.part_funcs_on_mls;
                  backward_term_4_align_on_mls = part_funcs.part_func_4_align + feature_score_sets.match_2_match_count;
                  logsumexp(&mut backward_term_4_align_on_mls, part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count);
                  logsumexp(&mut backward_term_4_align_on_mls, part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count);
                  backward_term_4_insert_on_mls = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut backward_term_4_insert_on_mls, part_funcs.part_func_4_insert + feature_score_sets.insert_extend_count);
                  logsumexp(&mut backward_term_4_insert_on_mls, part_funcs.part_func_4_insert_2 + feature_score_sets.insert_switch_count);
                  backward_term_4_insert_on_mls_2 = part_funcs.part_func_4_align + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut backward_term_4_insert_on_mls_2, part_funcs.part_func_4_insert + feature_score_sets.insert_switch_count);
                  logsumexp(&mut backward_term_4_insert_on_mls_2, part_funcs.part_func_4_insert_2 + feature_score_sets.insert_extend_count);
                }
                None => {}
              }
              match backward_tmp_part_func_set_mat_4_2loop.get(&pos_pair_2) {
                Some(part_funcs_4_2loop) => {
                  backward_term_4_align_4_2loop = part_funcs_4_2loop.part_func_4_align + feature_score_sets.match_2_match_count;
                  logsumexp(&mut backward_term_4_align_4_2loop, part_funcs_4_2loop.part_func_4_insert + feature_score_sets.match_2_insert_count);
                  logsumexp(&mut backward_term_4_align_4_2loop, part_funcs_4_2loop.part_func_4_insert_2 + feature_score_sets.match_2_insert_count);
                  backward_term_4_insert_4_2loop = part_funcs_4_2loop.part_func_4_align + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut backward_term_4_insert_4_2loop, part_funcs_4_2loop.part_func_4_insert + feature_score_sets.insert_extend_count);
                  logsumexp(&mut backward_term_4_insert_4_2loop, part_funcs_4_2loop.part_func_4_insert_2 + feature_score_sets.insert_switch_count);
                  backward_term_4_insert_4_2loop_2 = part_funcs_4_2loop.part_func_4_align + feature_score_sets.match_2_insert_count;
                  logsumexp(&mut backward_term_4_insert_4_2loop_2, part_funcs_4_2loop.part_func_4_insert + feature_score_sets.insert_switch_count);
                  logsumexp(&mut backward_term_4_insert_4_2loop_2, part_funcs_4_2loop.part_func_4_insert_2 + feature_score_sets.insert_extend_count);
                }, None => {},
              }
              let prob_coeff_4_hl = prob_coeff + hairpin_loop_score + hairpin_loop_score_2;
              let prob_coeff_4_ml = prob_coeff
                + multi_loop_closing_basepairing_score
                + multi_loop_closing_basepairing_score_2;
              match forward_tmp_part_func_set_mat.get(&pos_pair_4_loop_align) {
                Some(part_func_sets) => {
                  let ref part_funcs = part_func_sets.part_funcs_on_sa;
                  let mut loop_align_prob_4_hairpin_loop = NEG_INFINITY;
                  let term = prob_coeff_4_hl + loop_align_score + part_funcs.part_func_4_align + feature_score_sets.match_2_match_count + backward_term_4_align_on_sa;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_hairpin_loop, term);
                  let term = prob_coeff_4_hl + loop_align_score + part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + backward_term_4_align_on_sa;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_hairpin_loop, term);
                  let term = prob_coeff_4_hl + loop_align_score + part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + backward_term_4_align_on_sa;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_hairpin_loop, term);
                  if produces_struct_profs {
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_hl.0[long_u], loop_align_prob_4_hairpin_loop);
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_hl.1[long_v], loop_align_prob_4_hairpin_loop);
                  }
                  if produces_align_probs {
                    match sta_prob_mats.loop_align_prob_mat.get_mut(&pos_pair) {
                      Some(loop_align_prob) => {
                        logsumexp(loop_align_prob, loop_align_prob_4_hairpin_loop);
                      }
                      None => {
                        sta_prob_mats.loop_align_prob_mat.insert(pos_pair, loop_align_prob_4_hairpin_loop);
                      }
                    }
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_hairpin_loop);
                    logsumexp(&mut expected_feature_count_sets.align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_hairpin_loop);
                  }
                  let ref part_funcs = part_func_sets.part_funcs_on_sa_4_ml;
                  let mut loop_align_prob_4_multi_loop = NEG_INFINITY;
                  let term = prob_coeff_4_ml + loop_align_score + part_funcs.part_func_4_align + feature_score_sets.match_2_match_count + backward_term_4_align_4_ml + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_multi_loop, term);
                  let term = prob_coeff_4_ml + loop_align_score + part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + backward_term_4_align_4_ml + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_multi_loop, term);
                  let term = prob_coeff_4_ml + loop_align_score + part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + backward_term_4_align_4_ml + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_multi_loop, term);
                  if produces_struct_profs {
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u], loop_align_prob_4_multi_loop);
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v], loop_align_prob_4_multi_loop);
                  }
                  if produces_align_probs {
                    match sta_prob_mats.loop_align_prob_mat.get_mut(&pos_pair) {
                      Some(loop_align_prob) => {
                        logsumexp(loop_align_prob, loop_align_prob_4_multi_loop);
                      }
                      None => {
                        sta_prob_mats.loop_align_prob_mat.insert(pos_pair, loop_align_prob_4_multi_loop);
                      }
                    }
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_multi_loop);
                    logsumexp(&mut expected_feature_count_sets.align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_multi_loop);
                    logsumexp(&mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count, (2. as Prob).ln() + loop_align_prob_4_multi_loop);
                  }
                  let ref part_funcs = part_func_sets.part_funcs_4_first_bpas_on_mls;
                  let mut loop_align_prob_4_multi_loop = NEG_INFINITY;
                  let term = prob_coeff_4_ml
                    + loop_align_score
                    + part_funcs.part_func_4_align
                    + feature_score_sets.match_2_match_count
                    + backward_term_4_align_4_bpas_on_mls + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_multi_loop, term);
                  let term = prob_coeff_4_ml
                    + loop_align_score
                    + part_funcs.part_func_4_insert
                    + feature_score_sets.match_2_insert_count
                    + backward_term_4_align_4_bpas_on_mls + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_multi_loop, term);
                  let term = prob_coeff_4_ml
                    + loop_align_score
                    + part_funcs.part_func_4_insert_2
                    + feature_score_sets.match_2_insert_count
                    + backward_term_4_align_4_bpas_on_mls + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_multi_loop, term);
                  if produces_struct_profs {
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u], loop_align_prob_4_multi_loop);
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v], loop_align_prob_4_multi_loop);
                  }
                  if produces_align_probs {
                    match sta_prob_mats.loop_align_prob_mat.get_mut(&pos_pair) {
                      Some(loop_align_prob) => {
                        logsumexp(loop_align_prob, loop_align_prob_4_multi_loop);
                      }
                      None => {
                        sta_prob_mats.loop_align_prob_mat.insert(pos_pair, loop_align_prob_4_multi_loop);
                      }
                    }
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_multi_loop);
                    logsumexp(&mut expected_feature_count_sets.align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_multi_loop);
                    logsumexp(&mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count, (2. as Prob).ln() + loop_align_prob_4_multi_loop);
                  }
                  let ref part_funcs = part_func_sets.part_funcs_4_ml;
                  let mut loop_align_prob_4_multi_loop = NEG_INFINITY;
                  let term = prob_coeff_4_ml + loop_align_score + part_funcs.part_func_4_align + feature_score_sets.match_2_match_count + backward_term_4_align_on_mls + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_multi_loop, term);
                  let term = prob_coeff_4_ml + loop_align_score + part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + backward_term_4_align_on_mls + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_multi_loop, term);
                  let term = prob_coeff_4_ml + loop_align_score + part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + backward_term_4_align_on_mls + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_multi_loop, term);
                  if produces_struct_profs {
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u], loop_align_prob_4_multi_loop);
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v], loop_align_prob_4_multi_loop);
                  }
                  if produces_align_probs {
                    match sta_prob_mats.loop_align_prob_mat.get_mut(&pos_pair) {
                      Some(loop_align_prob) => {
                        logsumexp(loop_align_prob, loop_align_prob_4_multi_loop);
                      }
                      None => {
                        sta_prob_mats.loop_align_prob_mat.insert(pos_pair, loop_align_prob_4_multi_loop);
                      }
                    }
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_multi_loop);
                    logsumexp(&mut expected_feature_count_sets.align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_multi_loop);
                    logsumexp(&mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count, (2. as Prob).ln() + loop_align_prob_4_multi_loop);
                  }
                  let ref part_funcs = part_func_sets.part_funcs_on_sa;
                  let mut loop_align_prob_4_2loop = NEG_INFINITY;
                  let term = prob_coeff + loop_align_score + part_funcs.part_func_4_align + feature_score_sets.match_2_match_count + backward_term_4_align_4_2loop;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_2loop, term);
                  let term = prob_coeff + loop_align_score + part_funcs.part_func_4_insert + feature_score_sets.match_2_insert_count + backward_term_4_align_4_2loop;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_2loop, term);
                  let term = prob_coeff + loop_align_score + part_funcs.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + backward_term_4_align_4_2loop;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_2loop, term);
                  if produces_align_probs {
                    match sta_prob_mats.loop_align_prob_mat.get_mut(&pos_pair) {
                      Some(loop_align_prob) => {
                        logsumexp(loop_align_prob, loop_align_prob_4_2loop);
                      }
                      None => {
                        sta_prob_mats.loop_align_prob_mat.insert(pos_pair, loop_align_prob_4_2loop);
                      }
                    }
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_2loop);
                    logsumexp(&mut expected_feature_count_sets.align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_2loop);
                  }
                }
                None => {}
              }
              match forward_tmp_part_func_set_mat_4_2loop.get(&pos_pair_4_loop_align) {
                Some(part_funcs_4_2loop) => {
                  let mut loop_align_prob_4_2loop = NEG_INFINITY;
                  let term = prob_coeff + loop_align_score + part_funcs_4_2loop.part_func_4_align + feature_score_sets.match_2_match_count + backward_term_4_align_on_sa;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_2loop, term);
                  let term = prob_coeff + loop_align_score + part_funcs_4_2loop.part_func_4_insert + feature_score_sets.match_2_insert_count + backward_term_4_align_on_sa;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_2loop, term);
                  let term = prob_coeff + loop_align_score + part_funcs_4_2loop.part_func_4_insert_2 + feature_score_sets.match_2_insert_count + backward_term_4_align_on_sa;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut loop_align_prob_4_2loop, term);
                  if produces_align_probs {
                    match sta_prob_mats.loop_align_prob_mat.get_mut(&pos_pair) {
                      Some(loop_align_prob) => {
                        logsumexp(loop_align_prob, loop_align_prob_4_2loop);
                      }
                      None => {
                        sta_prob_mats.loop_align_prob_mat.insert(pos_pair, loop_align_prob_4_2loop);
                      }
                    }
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.loop_align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_2loop);
                    logsumexp(&mut expected_feature_count_sets.align_count_mat[dict_min_loop_align.0][dict_min_loop_align.1], loop_align_prob_4_2loop);
                  }
                }, None => {},
              }
              match forward_tmp_part_func_set_mat.get(&pos_pair_4_insert) {
                Some(part_func_sets) => {
                  let ref part_funcs = part_func_sets.part_funcs_on_sa;
                  let mut upp_4_hl = NEG_INFINITY;
                  let term = prob_coeff_4_hl
                    + insert_score
                    + feature_score_sets.match_2_insert_count
                    + part_funcs.part_func_4_align
                    + backward_term_4_insert_on_sa;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut upp_4_hl, term);
                  let term = prob_coeff_4_hl
                    + insert_score
                    + feature_score_sets.insert_extend_count
                    + part_funcs.part_func_4_insert
                    + backward_term_4_insert_on_sa;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
                  }
                  logsumexp(&mut upp_4_hl, term);
                  let term = prob_coeff_4_hl
                    + insert_score
                    + feature_score_sets.insert_switch_count
                    + part_funcs.part_func_4_insert_2
                    + backward_term_4_insert_on_sa;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
                  }
                  logsumexp(&mut upp_4_hl, term);
                  if produces_struct_profs {
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_hl.0[long_u], upp_4_hl);
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_counts[base], upp_4_hl);
                  }
                  let ref part_funcs = part_func_sets.part_funcs_on_sa_4_ml;
                  let mut upp_4_ml = NEG_INFINITY;
                  let term = prob_coeff_4_ml
                    + insert_score
                    + feature_score_sets.match_2_insert_count
                    + part_funcs.part_func_4_align
                    + backward_term_4_insert_4_ml + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  let term = prob_coeff_4_ml
                    + insert_score
                    + feature_score_sets.insert_extend_count
                    + part_funcs.part_func_4_insert
                    + backward_term_4_insert_4_ml + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  let term = prob_coeff_4_ml
                    + insert_score
                    + feature_score_sets.insert_switch_count
                    + part_funcs.part_func_4_insert_2
                    + backward_term_4_insert_4_ml + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  if produces_struct_profs {
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u], upp_4_ml);
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_counts[base], upp_4_ml);
                    logsumexp(&mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count, upp_4_ml);
                  }
                  let ref part_funcs = part_func_sets.part_funcs_4_first_bpas_on_mls;
                  let mut upp_4_ml = NEG_INFINITY;
                  let term = prob_coeff_4_ml
                    + insert_score
                    + feature_score_sets.match_2_insert_count
                    + part_funcs.part_func_4_align
                    + backward_term_4_insert_4_bpas_on_mls + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  let term = prob_coeff_4_ml
                    + insert_score
                    + feature_score_sets.insert_extend_count
                    + part_funcs.part_func_4_insert
                    + backward_term_4_insert_4_bpas_on_mls + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  let term = prob_coeff_4_ml
                    + insert_score
                    + feature_score_sets.insert_switch_count
                    + part_funcs.part_func_4_insert_2
                    + backward_term_4_insert_4_bpas_on_mls + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  if produces_struct_profs {
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u], upp_4_ml);
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_counts[base], upp_4_ml);
                    logsumexp(&mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count, upp_4_ml);
                  }
                  let ref part_funcs = part_func_sets.part_funcs_4_ml;
                  let mut upp_4_ml = NEG_INFINITY;
                  let term = prob_coeff_4_ml
                    + insert_score
                    + feature_score_sets.match_2_insert_count
                    + part_funcs.part_func_4_align
                    + backward_term_4_insert_on_mls + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  let term = prob_coeff_4_ml
                    + insert_score
                    + feature_score_sets.insert_extend_count
                    + part_funcs.part_func_4_insert
                    + backward_term_4_insert_on_mls + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  let term = prob_coeff_4_ml
                    + insert_score
                    + feature_score_sets.insert_switch_count
                    + part_funcs.part_func_4_insert_2
                    + backward_term_4_insert_on_mls + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  if produces_struct_profs {
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_ml.0[long_u], upp_4_ml);
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_counts[base], upp_4_ml);
                    logsumexp(&mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count, upp_4_ml);
                  }
                  let mut upp_4_2l = NEG_INFINITY;
                  let ref part_funcs = part_func_sets.part_funcs_on_sa;
                  let term = prob_coeff + insert_score + feature_score_sets.match_2_insert_count
                    + part_funcs.part_func_4_align
                    + backward_term_4_insert_4_2loop;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut upp_4_2l, term);
                  let term = prob_coeff + insert_score + feature_score_sets.insert_extend_count
                    + part_funcs.part_func_4_insert
                    + backward_term_4_insert_4_2loop;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
                  }
                  logsumexp(&mut upp_4_2l, term);
                  let term = prob_coeff + insert_score + feature_score_sets.insert_switch_count
                    + part_funcs.part_func_4_insert_2
                    + backward_term_4_insert_4_2loop;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
                  }
                  logsumexp(&mut upp_4_2l, term);
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_counts[base], upp_4_2l);
                  }
                }
                None => {}
              }
              match forward_tmp_part_func_set_mat_4_2loop.get(&pos_pair_4_insert) {
                Some(part_funcs_4_2loop) => {
                  let mut upp_4_2l = NEG_INFINITY;
                  let term = prob_coeff + insert_score + feature_score_sets.match_2_insert_count
                    + part_funcs_4_2loop.part_func_4_align
                    + backward_term_4_insert_on_sa;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut upp_4_2l, term);
                  let term = prob_coeff + insert_score + feature_score_sets.insert_extend_count
                    + part_funcs_4_2loop.part_func_4_insert
                    + backward_term_4_insert_on_sa;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
                  }
                  logsumexp(&mut upp_4_2l, term);
                  let term = prob_coeff + insert_score + feature_score_sets.insert_switch_count
                    + part_funcs_4_2loop.part_func_4_insert_2
                    + backward_term_4_insert_on_sa;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
                  }
                  logsumexp(&mut upp_4_2l, term);
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_counts[base], upp_4_2l);
                  }
                }, None => {},
              }
              match forward_tmp_part_func_set_mat.get(&pos_pair_4_insert_2) {
                Some(part_func_sets) => {
                  let ref part_funcs = part_func_sets.part_funcs_on_sa;
                  let mut upp_4_hl = NEG_INFINITY;
                  let term = prob_coeff_4_hl
                    + insert_score_2
                    + feature_score_sets.match_2_insert_count
                    + part_funcs.part_func_4_align
                    + backward_term_4_insert_on_sa_2;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut upp_4_hl, term);
                  let term = prob_coeff_4_hl
                    + insert_score_2
                    + feature_score_sets.insert_switch_count
                    + part_funcs.part_func_4_insert
                    + backward_term_4_insert_on_sa_2;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
                  }
                  logsumexp(&mut upp_4_hl, term);
                  let term = prob_coeff_4_hl
                    + insert_score_2
                    + feature_score_sets.insert_extend_count
                    + part_funcs.part_func_4_insert_2
                    + backward_term_4_insert_on_sa_2;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
                  }
                  logsumexp(&mut upp_4_hl, term);
                  if produces_struct_profs {
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_hl.1[long_v], upp_4_hl);
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_counts[base_2], upp_4_hl);
                  }
                  let ref part_funcs = part_func_sets.part_funcs_on_sa_4_ml;
                  let mut upp_4_ml = NEG_INFINITY;
                  let term = prob_coeff_4_ml
                    + insert_score_2
                    + feature_score_sets.match_2_insert_count
                    + part_funcs.part_func_4_align
                    + backward_term_4_insert_4_ml_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  let term = prob_coeff_4_ml
                    + insert_score_2
                    + feature_score_sets.insert_switch_count
                    + part_funcs.part_func_4_insert
                    + backward_term_4_insert_4_ml_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  let term = prob_coeff_4_ml
                    + insert_score_2
                    + feature_score_sets.insert_extend_count
                    + part_funcs.part_func_4_insert_2
                    + backward_term_4_insert_4_ml_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  if produces_struct_profs {
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v], upp_4_ml);
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_counts[base_2], upp_4_ml);
                    logsumexp(&mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count, upp_4_ml);
                  }
                  let ref part_funcs = part_func_sets.part_funcs_4_first_bpas_on_mls;
                  let mut upp_4_ml = NEG_INFINITY;
                  let term = prob_coeff_4_ml
                    + insert_score_2
                    + feature_score_sets.match_2_insert_count
                    + part_funcs.part_func_4_align
                    + backward_term_4_insert_4_bpas_on_mls_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  let term = prob_coeff_4_ml
                    + insert_score_2
                    + feature_score_sets.insert_switch_count
                    + part_funcs.part_func_4_insert
                    + backward_term_4_insert_4_bpas_on_mls_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  let term = prob_coeff_4_ml
                    + insert_score_2
                    + feature_score_sets.insert_extend_count
                    + part_funcs.part_func_4_insert_2
                    + backward_term_4_insert_4_bpas_on_mls_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  if produces_struct_profs {
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v], upp_4_ml);
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_counts[base_2], upp_4_ml);
                    logsumexp(&mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count, upp_4_ml);
                  }
                  let ref part_funcs = part_func_sets.part_funcs_4_ml;
                  let mut upp_4_ml = NEG_INFINITY;
                  let term = prob_coeff_4_ml
                    + insert_score_2
                    + feature_score_sets.match_2_insert_count
                    + part_funcs.part_func_4_align
                    + backward_term_4_insert_on_mls_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  let term = prob_coeff_4_ml
                    + insert_score_2
                    + feature_score_sets.insert_switch_count
                    + part_funcs.part_func_4_insert
                    + backward_term_4_insert_on_mls_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  let term = prob_coeff_4_ml
                    + insert_score_2
                    + feature_score_sets.insert_extend_count
                    + part_funcs.part_func_4_insert_2
                    + backward_term_4_insert_on_mls_2 + feature_score_sets.multi_loop_accessible_baseunpairing_count;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
                  }
                  logsumexp(&mut upp_4_ml, term);
                  if produces_struct_profs {
                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_ml.1[long_v], upp_4_ml);
                  }
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_counts[base_2], upp_4_ml);
                    logsumexp(&mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count, upp_4_ml);
                  }
                  let mut upp_4_2l = NEG_INFINITY;
                  let ref part_funcs = part_func_sets.part_funcs_on_sa;
                  let term = prob_coeff + insert_score_2 + feature_score_sets.match_2_insert_count
                    + part_funcs.part_func_4_align
                    + backward_term_4_insert_4_2loop_2;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut upp_4_2l, term);
                  let term = prob_coeff + insert_score_2 + feature_score_sets.insert_switch_count
                    + part_funcs.part_func_4_insert
                    + backward_term_4_insert_4_2loop_2;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
                  }
                  logsumexp(&mut upp_4_2l, term);
                  let term = prob_coeff + insert_score_2 + feature_score_sets.insert_extend_count
                    + part_funcs.part_func_4_insert_2
                    + backward_term_4_insert_4_2loop_2;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
                  }
                  logsumexp(&mut upp_4_2l, term);
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_counts[base_2], upp_4_2l);
                  }
                }
                None => {}
              }
              match forward_tmp_part_func_set_mat_4_2loop.get(&pos_pair_4_insert_2) {
                Some(part_funcs_4_2loop) => {
                  let mut upp_4_2l = NEG_INFINITY;
                  let term = prob_coeff + insert_score_2 + feature_score_sets.match_2_insert_count
                    + part_funcs_4_2loop.part_func_4_align
                    + backward_term_4_insert_on_sa_2;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                  }
                  logsumexp(&mut upp_4_2l, term);
                  let term = prob_coeff + insert_score_2 + feature_score_sets.insert_switch_count
                    + part_funcs_4_2loop.part_func_4_insert
                    + backward_term_4_insert_on_sa_2;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_switch_count, term);
                  }
                  logsumexp(&mut upp_4_2l, term);
                  let term = prob_coeff + insert_score_2 + feature_score_sets.insert_extend_count
                    + part_funcs_4_2loop.part_func_4_insert_2
                    + backward_term_4_insert_on_sa_2;
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_extend_count, term);
                  }
                  logsumexp(&mut upp_4_2l, term);
                  if trains_score_params {
                    logsumexp(&mut expected_feature_count_sets.insert_counts[base_2], upp_4_2l);
                  }
                }, None => {},
              }
            }
            None => {}
          }
        }
      }
    }
    if produces_struct_profs {
      for upp in sta_prob_mats.upp_mat_pair_4_hl.0.iter_mut() {
        *upp = expf(*upp);
      }
      for upp in sta_prob_mats.upp_mat_pair_4_hl.1.iter_mut() {
        *upp = expf(*upp);
      }
      for upp in sta_prob_mats.upp_mat_pair_4_2l.0.iter_mut() {
        *upp = expf(*upp);
      }
      for upp in sta_prob_mats.upp_mat_pair_4_2l.1.iter_mut() {
        *upp = expf(*upp);
      }
      for upp in sta_prob_mats.upp_mat_pair_4_ml.0.iter_mut() {
        *upp = expf(*upp);
      }
      for upp in sta_prob_mats.upp_mat_pair_4_ml.1.iter_mut() {
        *upp = expf(*upp);
      }
      for upp in sta_prob_mats.upp_mat_pair_4_el.0.iter_mut() {
        *upp = expf(*upp);
      }
      for upp in sta_prob_mats.upp_mat_pair_4_el.1.iter_mut() {
        *upp = expf(*upp);
      }
      for bpp in sta_prob_mats.bpp_mat_pair_2.0.iter_mut() {
        *bpp = expf(*bpp);
      }
      for bpp in sta_prob_mats.bpp_mat_pair_2.1.iter_mut() {
        *bpp = expf(*bpp);
      }
    }
    if produces_align_probs {
      for loop_align_prob in sta_prob_mats.loop_align_prob_mat.values_mut() {
        *loop_align_prob = expf(*loop_align_prob);
      }
      for basepair_align_prob in sta_prob_mats.basepair_align_prob_mat.values_mut() {
        *basepair_align_prob = expf(*basepair_align_prob);
      }
    }
    if trains_score_params {
      for count in expected_feature_count_sets.hairpin_loop_length_counts.iter_mut() {
        *count = expf(*count);
      }
      for count in expected_feature_count_sets.bulge_loop_length_counts.iter_mut() {
        *count = expf(*count);
      }
      for count in expected_feature_count_sets.interior_loop_length_counts.iter_mut() {
        *count = expf(*count);
      }
      for count in expected_feature_count_sets.interior_loop_length_counts_symm.iter_mut() {
        *count = expf(*count);
      }
      for count in expected_feature_count_sets.interior_loop_length_counts_asymm.iter_mut() {
        *count = expf(*count);
      }
      for count_3d_mat in expected_feature_count_sets.stack_count_mat.iter_mut() {
        for count_2d_mat in count_3d_mat.iter_mut() {
          for counts in count_2d_mat.iter_mut() {
            for count in counts.iter_mut() {
              *count = expf(*count);
            }
          }
        }
      }
      for count_3d_mat in expected_feature_count_sets.terminal_mismatch_count_mat.iter_mut() {
        for count_2d_mat in count_3d_mat.iter_mut() {
          for counts in count_2d_mat.iter_mut() {
            for count in counts.iter_mut() {
              *count = expf(*count);
            }
          }
        }
      }
      for count_2d_mat in expected_feature_count_sets.left_dangle_count_mat.iter_mut() {
        for counts in count_2d_mat.iter_mut() {
          for count in counts.iter_mut() {
            *count = expf(*count);
          }
        }
      }
      for count_2d_mat in expected_feature_count_sets.right_dangle_count_mat.iter_mut() {
        for counts in count_2d_mat.iter_mut() {
          for count in counts.iter_mut() {
            *count = expf(*count);
          }
        }
      }
      for counts in expected_feature_count_sets.interior_loop_length_count_mat_explicit.iter_mut() {
        for count in counts.iter_mut() {
          *count = expf(*count);
        }
      }
      for count in expected_feature_count_sets.bulge_loop_0x1_length_counts.iter_mut() {
        *count = expf(*count);
      }
      for counts in expected_feature_count_sets.interior_loop_1x1_length_count_mat.iter_mut() {
        for count in counts.iter_mut() {
          *count = expf(*count);
        }
      }
      for counts in expected_feature_count_sets.helix_end_count_mat.iter_mut() {
        for count in counts.iter_mut() {
          *count = expf(*count);
        }
      }
      for counts in expected_feature_count_sets.base_pair_count_mat.iter_mut() {
        for count in counts.iter_mut() {
          *count = expf(*count);
        }
      }
      expected_feature_count_sets.multi_loop_base_count = expf(expected_feature_count_sets.multi_loop_base_count);
      expected_feature_count_sets.multi_loop_basepairing_count = expf(expected_feature_count_sets.multi_loop_basepairing_count);
      expected_feature_count_sets.multi_loop_accessible_baseunpairing_count = expf(expected_feature_count_sets.multi_loop_accessible_baseunpairing_count);
      expected_feature_count_sets.external_loop_accessible_basepairing_count = expf(expected_feature_count_sets.external_loop_accessible_basepairing_count);
      expected_feature_count_sets.external_loop_accessible_baseunpairing_count = expf(expected_feature_count_sets.external_loop_accessible_baseunpairing_count);
      for count_3d_mat in expected_feature_count_sets.basepair_align_count_mat.iter_mut() {
        for count_2d_mat in count_3d_mat.iter_mut() {
          for counts in count_2d_mat.iter_mut() {
            for count in counts.iter_mut() {
              *count = expf(*count);
            }
          }
        }
      }
      for counts in expected_feature_count_sets.loop_align_count_mat.iter_mut() {
        for count in counts.iter_mut() {
          *count = expf(*count);
        }
      }
      expected_feature_count_sets.match_2_match_count = expf(expected_feature_count_sets.match_2_match_count);
      expected_feature_count_sets.match_2_insert_count = expf(expected_feature_count_sets.match_2_insert_count);
      expected_feature_count_sets.insert_extend_count = expf(expected_feature_count_sets.insert_extend_count);
      expected_feature_count_sets.insert_switch_count = expf(expected_feature_count_sets.insert_switch_count);
      for count in expected_feature_count_sets.insert_counts.iter_mut() {
        *count = expf(*count);
      }
      for counts in expected_feature_count_sets.align_count_mat.iter_mut() {
        for count in counts.iter_mut() {
          *count = expf(*count);
        }
      }
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
  produces_struct_profs: bool,
) -> PctStaProbMats<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord,
{
  let weight = 1. / (num_of_rnas - 1) as Prob;
  let mut pct_prob_mats = PctStaProbMats::new(upp_mat_len);
  pct_prob_mats.bpp_mat_on_ss = bpp_mat.clone();
  for rna_id_2 in 0 .. num_of_rnas {
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
    let ref_2_bpp_mat_2 = if rna_id < rna_id_2 {
      &ref_2_prob_mats.bpp_mat_pair_2.0
    } else {
      &ref_2_prob_mats.bpp_mat_pair_2.1
    };
    for (i, &bpp) in ref_2_bpp_mat_2.iter().enumerate() {
      let weighted_bpp = weight * bpp;
      pct_prob_mats.bpp_mat_2[i] += weighted_bpp;
    }
    if produces_struct_profs {
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

pub fn is_min_gap_ok<T>(pos_pair: &PosPair<T>, pos_quadruple: &PosQuadruple<T>, max_gap_num: T) -> bool
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
  accessible_bp_shared_score_with_dangle: FeatureCount,
) -> FeatureCount {
  let mismatch_pair = (seq[pos_pair.1 + 1], seq[pos_pair.0 - 1]);
  if mismatch_pair.0 == PSEUDO_BASE || mismatch_pair.1 == PSEUDO_BASE {
    NEG_INFINITY
  } else {
    feature_score_sets.multi_loop_basepairing_count
      + accessible_bp_shared_score_with_dangle
  }
}

pub fn get_consprob_external_loop_accessible_basepairing_score(
  feature_score_sets: &FeatureCountSets,
  accessible_bp_shared_score_with_dangle: FeatureCount,
) -> FeatureCount {
  feature_score_sets.external_loop_accessible_basepairing_count
    + accessible_bp_shared_score_with_dangle
}

pub fn get_consprob_hairpin_loop_score(
  feature_score_sets: &FeatureCountSets,
  seq: SeqSlice,
  pos_pair: &(usize, usize),
) -> FeatureCount {
  let hairpin_loop_len = pos_pair.1 - pos_pair.0 - 1;
  if hairpin_loop_len > CONSPROB_MAX_HAIRPIN_LOOP_LEN || hairpin_loop_len < CONSPROB_MIN_HAIRPIN_LOOP_LEN {
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
  accessible_bp_shared_subscore: FeatureCount,
  helix_end_score: FeatureCount,
  terminal_mismatch_score: FeatureCount,
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
  let loop_len_pair = (
    pos_pair_accessible.0 - pos_pair_closing_loop.0 - 1,
    pos_pair_closing_loop.1 - pos_pair_accessible.1 - 1,
  );
  
  if is_stack {
    feature_score_sets.base_pair_count_mat[base_pair_accessible.0][base_pair_accessible.1]
      + feature_score_sets.stack_count_mat[base_pair_closing_loop.0][base_pair_closing_loop.1]
        [base_pair_accessible.0][base_pair_accessible.1]
  } else {
    accessible_bp_shared_subscore
      + feature_score_sets.helix_end_count_mat[base_pair_closing_loop.0]
        [base_pair_closing_loop.1]
      + helix_end_score
      + feature_score_sets.terminal_mismatch_count_mat[base_pair_closing_loop.0][base_pair_closing_loop.1][mismatch_pair_1.0][mismatch_pair_1.1]
      + terminal_mismatch_score + if is_bulge_loop {
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

pub fn get_dict_min_align(align: &BasePair) -> BasePair {
  get_dict_min_loop_align(align)
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

pub fn get_sparse_pos_sets<T>(bpp_mat_pair: &ProbMatPair<T>, max_gap_num_4_il: T, pseudo_pos_quadruple: &PosQuadruple<T>) -> (PosPairMatSet<T>, PosPairMatSet<T>, PosQuadrupleMat<T>)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send,
{
  let mut forward_pos_pair_mat_set = PosPairMatSet::<T>::default();
  let mut backward_pos_pair_mat_set = PosPairMatSet::<T>::default();
  let mut pos_quadruple_mat = PosQuadrupleMat::<T>::default();
  for pos_pair in bpp_mat_pair.0.keys() {
    for pos_pair_2 in bpp_mat_pair.1.keys() {
      let forward_pos_pair = (pos_pair.0, pos_pair_2.0);
      let backward_pos_pair = (pos_pair.1, pos_pair_2.1);
      if !is_min_gap_ok(&forward_pos_pair, &pseudo_pos_quadruple, max_gap_num_4_il) {
        continue;
      }
      if !is_min_gap_ok(&backward_pos_pair, &pseudo_pos_quadruple, max_gap_num_4_il) {
        continue;
      }
      if !forward_pos_pair_mat_set.contains_key(&forward_pos_pair) {
        forward_pos_pair_mat_set.insert(forward_pos_pair, PosPairMat::<T>::default());
      }
      forward_pos_pair_mat_set.get_mut(&forward_pos_pair).unwrap().insert(backward_pos_pair);
      if !backward_pos_pair_mat_set.contains_key(&backward_pos_pair) {
        backward_pos_pair_mat_set.insert(backward_pos_pair, PosPairMat::<T>::default());
      }
      backward_pos_pair_mat_set.get_mut(&backward_pos_pair).unwrap().insert(forward_pos_pair);
      pos_quadruple_mat.insert((pos_pair.0, pos_pair.1, pos_pair_2.0, pos_pair_2.1));
    }
  }
  (forward_pos_pair_mat_set, backward_pos_pair_mat_set, pos_quadruple_mat)
}

pub fn consprob<T>(
  thread_pool: &mut Pool,
  fasta_records: &FastaRecords,
  min_bpp: Prob,
  offset_4_max_gap_num: T,
  produces_struct_profs: bool,
  produces_align_probs: bool,
) -> (ProbMatSets<T>, AlignProbMatSetsWithRnaIdPairs<T>)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let feature_score_sets = FeatureCountSets::load_trained_score_params();
  let num_of_fasta_records = fasta_records.len();
  let mut bpp_mats = vec![SparseProbMat::<T>::new(); num_of_fasta_records];
  let mut sparse_bpp_mats = vec![SparseProbMat::<T>::new(); num_of_fasta_records];
  let mut max_bp_spans = vec![T::zero(); num_of_fasta_records];
  let mut bp_score_param_set_seqs = vec![BpScoreParamSets::<T>::new(); num_of_fasta_records];
  let ref ref_2_feature_score_sets = feature_score_sets;
  thread_pool.scoped(|scope| {
    for (bpp_mat, sparse_bpp_mat, max_bp_span, fasta_record, bp_score_param_sets) in multizip((
      bpp_mats.iter_mut(),
      sparse_bpp_mats.iter_mut(),
      max_bp_spans.iter_mut(),
      fasta_records.iter(),
      bp_score_param_set_seqs.iter_mut(),
    )) {
      let seq_len = fasta_record.seq.len();
      scope.execute(move || {
        *bpp_mat = mccaskill_algo(&fasta_record.seq[1..seq_len - 1], false).0;
        *sparse_bpp_mat = remove_small_bpps_from_bpp_mat::<T>(bpp_mat, min_bpp);
        *max_bp_span = get_max_bp_span::<T>(sparse_bpp_mat);
        *bp_score_param_sets = BpScoreParamSets::<T>::set_curr_params(ref_2_feature_score_sets, &fasta_record.seq[..], sparse_bpp_mat);
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
      let bp_score_param_set_pair = (
        &bp_score_param_set_seqs[rna_id_pair.0],
        &bp_score_param_set_seqs[rna_id_pair.1],
      );
      let (forward_pos_pair_mat_set, backward_pos_pair_mat_set, pos_quadruple_mat) = get_sparse_pos_sets(&bpp_mat_pair, max_gap_num_4_il, &pseudo_pos_quadruple);
      let ref ref_2_feature_score_sets = feature_score_sets;
      scope.execute(move || {
        *prob_mats = io_algo_4_prob_mats::<T>(
          &seq_pair,
          ref_2_feature_score_sets,
          &max_bp_span_pair,
          max_gap_num,
          max_gap_num_4_il,
          produces_struct_profs,
          false,
          &mut FeatureCountSets::new(NEG_INFINITY),
          &forward_pos_pair_mat_set,
          &backward_pos_pair_mat_set,
          &pos_quadruple_mat,
          &bp_score_param_set_pair,
          produces_align_probs,
        ).0;
      });
    }
  });
  let ref ref_2_prob_mats_with_rna_id_pairs = prob_mats_with_rna_id_pairs;
  let mut prob_mat_sets = vec![PctStaProbMats::<T>::origin(); num_of_fasta_records];
  thread_pool.scoped(|scope| {
    for (rna_id, prob_mats, bpp_mat) in multizip((
      0..num_of_fasta_records,
      prob_mat_sets.iter_mut(),
      bpp_mats.iter_mut(),
    )) {
      let seq_len = fasta_records[rna_id].seq.len();
      scope.execute(move || {
        *prob_mats = pct_of_prob_mats::<T>(
          ref_2_prob_mats_with_rna_id_pairs,
          rna_id,
          num_of_fasta_records,
          bpp_mat,
          seq_len,
          produces_struct_profs,
        );
      });
    }
  });
  let mut align_prob_mat_sets_with_rna_id_pairs = AlignProbMatSetsWithRnaIdPairs::<T>::default();
  if produces_align_probs {
    for rna_id_1 in 0 .. num_of_fasta_records {
      for rna_id_2 in rna_id_1 + 1 .. num_of_fasta_records {
        let rna_id_pair = (rna_id_1, rna_id_2);
        let ref prob_mats = prob_mats_with_rna_id_pairs[&rna_id_pair];
        let mut align_prob_mats = AlignProbMats::<T>::new();
        align_prob_mats.loop_align_prob_mat = prob_mats.loop_align_prob_mat.clone();
        align_prob_mats.basepair_align_prob_mat = prob_mats.basepair_align_prob_mat.clone();
        align_prob_mats.align_prob_mat = align_prob_mats.loop_align_prob_mat.clone();
        for (pos_quadruple, &bpap) in &align_prob_mats.basepair_align_prob_mat {
          let pos_pair = (pos_quadruple.0, pos_quadruple.2);
          match align_prob_mats.align_prob_mat.get_mut(&pos_pair) {
            Some(bap) => {
              *bap += bpap;
            }, None => {
              align_prob_mats.align_prob_mat.insert(pos_pair, bpap);
            }
          }
          let pos_pair = (pos_quadruple.1, pos_quadruple.3);
          match align_prob_mats.align_prob_mat.get_mut(&pos_pair) {
            Some(bap) => {
              *bap += bpap;
            }, None => {
              align_prob_mats.align_prob_mat.insert(pos_pair, bpap);
            }
          }
        }
        align_prob_mat_sets_with_rna_id_pairs.insert(rna_id_pair, align_prob_mats);
      }
    }
  }
  (prob_mat_sets, align_prob_mat_sets_with_rna_id_pairs)
}

pub fn constrain<'a, T>(
  thread_pool: &mut Pool,
  train_data: &mut TrainData<T>,
  offset_4_max_gap_num: T,
  output_file_path: &Path,
)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let mut feature_score_sets = FeatureCountSets::new(0.);
  // feature_score_sets.rand_init();
  feature_score_sets.transfer();
  for train_datum in train_data.iter_mut() {
    train_datum.set_curr_params(&feature_score_sets);
  }
  let mut old_feature_score_sets = feature_score_sets.clone();
  let mut old_cost = INFINITY;
  let mut costs = Probs::new();
  let mut count = 0;
  let mut regularizers = Regularizers::from(vec![1.; feature_score_sets.get_len()]);
  let num_of_data = train_data.len() as FeatureCount;
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
        let ref max_bp_span_pair = train_datum.max_bp_span_pair;
        let ref mut expected_feature_count_sets = train_datum.expected_feature_count_sets;
        let ref mut part_func = train_datum.part_func;
        let ref forward_pos_pair_mat_set = train_datum.forward_pos_pair_mat_set;
        let ref backward_pos_pair_mat_set = train_datum.backward_pos_pair_mat_set;
        let ref pos_quadruple_mat = train_datum.pos_quadruple_mat;
        let bp_score_param_set_pair = (&train_datum.bp_score_param_set_pair.0, &train_datum.bp_score_param_set_pair.1);
        scope.execute(move || {
          *part_func = io_algo_4_prob_mats::<T>(
            &seq_pair,
            ref_2_feature_score_sets,
            max_bp_span_pair,
            max_gap_num,
            max_gap_num_4_il,
            false,
            true,
            expected_feature_count_sets,
            forward_pos_pair_mat_set,
            backward_pos_pair_mat_set,
            pos_quadruple_mat,
            &bp_score_param_set_pair,
            false,
          ).1;
        });
      }
    });
    feature_score_sets.update(&train_data, &mut regularizers);
    for train_datum in train_data.iter_mut() {
      train_datum.set_curr_params(&feature_score_sets);
    }
    let cost = feature_score_sets.get_cost(&train_data[..], &regularizers);
    if old_cost.is_finite() && (old_cost - cost) / num_of_data <= LEARNING_TOLERANCE {
      feature_score_sets = old_feature_score_sets.clone();
      break;
    }
    costs.push(cost);
    old_feature_score_sets = feature_score_sets.clone();
    old_cost = cost;
    println!("Epoch {} finished (current cost = {})", count + 1, cost);
    count += 1;
  }
  write_feature_score_sets_trained(&feature_score_sets);
  write_costs(&costs, output_file_path);
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
  f.match_2_match_count = feature_counts[offset];
  offset += 1;
  f.match_2_insert_count = feature_counts[offset];
  offset += 1;
  f.insert_extend_count = feature_counts[offset];
  offset += 1;
  f.insert_switch_count = feature_counts[offset];
  offset += 1;
  let len = f.insert_counts.len();
  for i in 0 .. len {
    f.insert_counts[i] = feature_counts[offset + i];
  }
  offset += len;
  let len = f.align_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      f.align_count_mat[i][j] = feature_counts[offset + i * len + j];
    }
  }
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
  feature_counts[offset] = f.match_2_match_count;
  offset += 1;
  feature_counts[offset] = f.match_2_insert_count;
  offset += 1;
  feature_counts[offset] = f.insert_extend_count;
  offset += 1;
  feature_counts[offset] = f.insert_switch_count;
  offset += 1;
  let len = f.insert_counts.len();
  for i in 0 .. len {
    feature_counts[offset + i] = f.insert_counts[i];
  }
  offset += len;
  let len = f.align_count_mat.len();
  for i in 0 .. len {
    for j in 0 .. len {
      feature_counts[offset + i * len + j] = f.align_count_mat[i][j];
    }
  }
  Array::from(feature_counts)
}

pub fn convert_feature_counts_2_bfgs_feature_counts(feature_counts: &FeatureCounts) -> BfgsFeatureCounts {
  let vec: Vec<BfgsFeatureCount> = feature_counts.to_vec().iter().map(|x| *x as BfgsFeatureCount).collect();
  BfgsFeatureCounts::from(vec)
}

pub fn convert_bfgs_feature_counts_2_feature_counts(feature_counts: &BfgsFeatureCounts) -> FeatureCounts {
  let vec: Vec<FeatureCount> = feature_counts.to_vec().iter().map(|x| *x as FeatureCount).collect();
  FeatureCounts::from(vec)
}

pub fn get_regularizer(group_size: usize, squared_sum: FeatureCount) -> Regularizer {
  (group_size as FeatureCount / 2. + GAMMA_DIST_ALPHA) / (squared_sum / 2. + GAMMA_DIST_BETA)
}

pub fn write_feature_score_sets_trained(feature_score_sets: &FeatureCountSets) {
  let mut writer_2_trained_feature_score_sets_file = BufWriter::new(File::create(TRAINED_FEATURE_SCORE_SETS_FILE_PATH).unwrap());
  let mut buf_4_writer_2_trained_feature_score_sets_file = String::from("use FeatureCountSets;\nimpl FeatureCountSets {\npub fn load_trained_score_params() -> FeatureCountSets {\nFeatureCountSets {\nhairpin_loop_length_counts: ");
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
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nmatch_2_match_count: ", &feature_score_sets.loop_align_count_mat));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nmatch_2_insert_count: ", feature_score_sets.match_2_match_count));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninsert_extend_count: ", feature_score_sets.match_2_insert_count));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninsert_switch_count: ", feature_score_sets.insert_extend_count));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninsert_counts: ", feature_score_sets.insert_switch_count));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nalign_count_mat: ", feature_score_sets.insert_counts));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nhairpin_loop_length_counts_cumulative: ", feature_score_sets.align_count_mat));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\nbulge_loop_length_counts_cumulative: ", &feature_score_sets.hairpin_loop_length_counts_cumulative));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninterior_loop_length_counts_cumulative: ", &feature_score_sets.bulge_loop_length_counts_cumulative));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninterior_loop_length_counts_symm_cumulative: ", &feature_score_sets.interior_loop_length_counts_cumulative));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},\ninterior_loop_length_counts_asymm_cumulative: ", &feature_score_sets.interior_loop_length_counts_symm_cumulative));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!("{:?},", &feature_score_sets.interior_loop_length_counts_asymm_cumulative));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&String::from("\n}\n}\n}"));
  let _ = writer_2_trained_feature_score_sets_file.write_all(buf_4_writer_2_trained_feature_score_sets_file.as_bytes());
}

pub fn write_costs(costs: &Probs, output_file_path: &Path) {
  let mut writer_2_output_file = BufWriter::new(File::create(output_file_path).unwrap());
  let mut buf_4_writer_2_output_file = String::new();
  for cost in costs {
    buf_4_writer_2_output_file.push_str(&format!("{}\n", cost));
  }
  let _ = writer_2_output_file.write_all(buf_4_writer_2_output_file.as_bytes());
}

pub fn write_prob_mat_sets<T>(
  output_dir_path: &Path,
  prob_mat_sets: &ProbMatSets<T>,
  produces_struct_profs: bool,
  align_prob_mat_sets_with_rna_id_pairs: &AlignProbMatSetsWithRnaIdPairs<T>,
  produces_align_probs: bool,
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
  let bpp_mat_file_path_2 = output_dir_path.join(BPP_MAT_FILE_NAME_2);
  let mut writer_2_bpp_mat_file_2 = BufWriter::new(File::create(bpp_mat_file_path_2).unwrap());
  let mut buf_4_writer_2_bpp_mat_file_2 = String::new();
  for (rna_id, prob_mats) in prob_mat_sets.iter().enumerate() {
    let seq_len = prob_mats.bpp_mat_2.len();
    let mut buf_4_rna_id = format!("\n\n>{}\n", rna_id);
    for (i, &bpp) in prob_mats.bpp_mat_2.iter().enumerate() {
      if i == 0 || i == seq_len - 1 {continue;}
      buf_4_rna_id.push_str(&format!("{},{} ", i - 1, bpp));
    }
    buf_4_writer_2_bpp_mat_file_2.push_str(&buf_4_rna_id);
  }
  let _ = writer_2_bpp_mat_file_2.write_all(buf_4_writer_2_bpp_mat_file_2.as_bytes());
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
  if produces_struct_profs {
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
  if produces_align_probs {
    let loop_align_prob_mat_file_path = output_dir_path.join(LOOP_ALIGN_PROB_MAT_FILE_NAME);
    let basepair_align_prob_mat_file_path = output_dir_path.join(BASEPAIR_ALIGN_PROB_MAT_FILE_NAME);
    let align_prob_mat_file_path = output_dir_path.join(ALIGN_PROB_MAT_FILE_NAME);
    let mut writer_2_loop_align_prob_mat_file = BufWriter::new(File::create(loop_align_prob_mat_file_path).unwrap());
    let mut writer_2_basepair_align_prob_mat_file = BufWriter::new(File::create(basepair_align_prob_mat_file_path).unwrap());
    let mut writer_2_align_prob_mat_file = BufWriter::new(File::create(align_prob_mat_file_path).unwrap());
    let mut buf_4_writer_2_loop_align_prob_mat_file = String::new();
    let mut buf_4_writer_2_basepair_align_prob_mat_file = String::new();
    let mut buf_4_writer_2_align_prob_mat_file = String::new();
    for (rna_id_pair, align_prob_mats) in align_prob_mat_sets_with_rna_id_pairs.iter() {
      let mut buf_4_rna_id_pair = format!("\n\n>{},{}\n", rna_id_pair.0, rna_id_pair.1);
      for (&(i, j), &loop_align_prob) in align_prob_mats.loop_align_prob_mat.iter() {
        buf_4_rna_id_pair.push_str(&format!("{},{},{} ", i - T::one(), j - T::one(), loop_align_prob));
      }
      buf_4_writer_2_loop_align_prob_mat_file.push_str(&buf_4_rna_id_pair);
      let mut buf_4_rna_id_pair = format!("\n\n>{},{}\n", rna_id_pair.0, rna_id_pair.1);
      for (&(i, j, k, l), &basepair_align_prob) in align_prob_mats.basepair_align_prob_mat.iter() {
        buf_4_rna_id_pair.push_str(&format!("{},{},{},{},{} ", i - T::one(), j - T::one(), k - T::one(), l - T::one(), basepair_align_prob));
      }
      buf_4_writer_2_basepair_align_prob_mat_file.push_str(&buf_4_rna_id_pair);
      let mut buf_4_rna_id_pair = format!("\n\n>{},{}\n", rna_id_pair.0, rna_id_pair.1);
      for (&(i, j), &align_prob) in align_prob_mats.align_prob_mat.iter() {
        buf_4_rna_id_pair.push_str(&format!("{},{},{} ", i - T::one(), j - T::one(), align_prob));
      }
      buf_4_writer_2_align_prob_mat_file.push_str(&buf_4_rna_id_pair);
    }
    let _ = writer_2_loop_align_prob_mat_file.write_all(buf_4_writer_2_loop_align_prob_mat_file.as_bytes());
    let _ = writer_2_basepair_align_prob_mat_file.write_all(buf_4_writer_2_basepair_align_prob_mat_file.as_bytes());
    let _ = writer_2_align_prob_mat_file.write_all(buf_4_writer_2_align_prob_mat_file.as_bytes());
  }
}

pub fn write_readme(output_dir_path: &Path, readme_contents: &String) {
  let readme_file_path = output_dir_path.join(README_FILE_NAME);
  let mut writer_2_readme_file = BufWriter::new(File::create(readme_file_path).unwrap());
  let buf_4_writer_2_readme_file = String::from(readme_contents);
  let _ = writer_2_readme_file.write_all(buf_4_writer_2_readme_file.as_bytes());
}
