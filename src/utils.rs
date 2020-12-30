pub use rna_algos::mccaskill_algo::*;
pub use rna_algos::utils::*;

pub type BaScoreMat = HashMap<BasePair, FreeEnergy>;
pub type BpaScoreMat = HashMap<(BasePair, BasePair), FreeEnergy>;
pub type InsertScores = [FreeEnergy; NUM_OF_BASES];

pub const PSEUDO_BASE: Base = U + 1 as Base;
