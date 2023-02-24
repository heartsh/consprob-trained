extern crate consprob_trained;

use consprob_trained::*;
use std::env;

fn main() {
  let args = env::args().collect::<Args>();
  let program_name = args[0].clone();
  let mut opts = Options::new();
  opts.reqopt(
    "i",
    "input_dir",
    "An input directory path containing RNA structural alignments in FASTA format",
    "STR",
  );
  opts.reqopt(
    "o",
    "output_file_path",
    "An output file path to record intermediate training logs",
    "STR",
  );
  opts.optopt(
    "",
    "min_basepair_prob",
    &format!("A minimum base-pairing probability (Use {DEFAULT_BASEPAIR_PROB_TRAIN} by default)"),
    "FLOAT",
  );
  opts.optopt(
    "",
    "min_match_prob",
    &format!("A minimum matching probability (Use {DEFAULT_MATCH_PROB_TRAIN} by default)"),
    "FLOAT",
  );
  opts.optopt(
    "",
    "learning_tolerance",
    &format!(
      "The lower threshold of training accuracy change to quit learning (Use {DEFAULT_LEARNING_TOLERANCE} by default)"
    ),
    "FLOAT",
  );
  opts.optopt(
    "t",
    "num_threads",
    "The number of threads in multithreading (Use all the threads of this computer by default)",
    "UINT",
  );
  opts.optflag(
    "r",
    "enables_randinit",
    "Enable the random initialization of alignment scoring parameters to be trained",
  );
  opts.optflag("h", "help", "Print a help menu");
  let matches = match opts.parse(&args[1..]) {
    Ok(x) => x,
    Err(x) => {
      print_program_usage(&program_name, &opts);
      panic!("{}", x.to_string())
    }
  };
  if matches.opt_present("h") {
    print_program_usage(&program_name, &opts);
    return;
  }
  let input_dir_path = matches.opt_str("i").unwrap();
  let input_dir_path = Path::new(&input_dir_path);
  let min_basepair_prob = if matches.opt_present("min_basepair_prob") {
    matches
      .opt_str("min_basepair_prob")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_BASEPAIR_PROB_TRAIN
  };
  let min_match_prob = if matches.opt_present("min_match_prob") {
    matches.opt_str("min_match_prob").unwrap().parse().unwrap()
  } else {
    DEFAULT_MATCH_PROB_TRAIN
  };
  let learning_tolerance = if matches.opt_present("learning_tolerance") {
    matches
      .opt_str("learning_tolerance")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_LEARNING_TOLERANCE
  };
  let enables_randinit = matches.opt_present("r");
  let num_threads = if matches.opt_present("t") {
    matches.opt_str("t").unwrap().parse().unwrap()
  } else {
    num_cpus::get() as NumThreads
  };
  println!("# threads = {num_threads}");
  let output_file_path = matches.opt_str("o").unwrap();
  let output_file_path = Path::new(&output_file_path);
  print_train_info(&AlignfoldScores::new(0.));
  let entries: Vec<DirEntry> = read_dir(input_dir_path)
    .unwrap()
    .map(|x| x.unwrap())
    .collect();
  let num_entries = entries.len();
  let mut train_data = vec![TrainDatum::origin(); num_entries];
  let mut thread_pool = Pool::new(num_threads);
  let mut align_scores = AlignScores::new(0.);
  align_scores.transfer();
  thread_pool.scoped(|x| {
    let y = &align_scores;
    for (z, a) in entries.iter().zip(train_data.iter_mut()) {
      x.execute(move || {
        *a = TrainDatum::<u16>::new(&z.path(), min_basepair_prob, min_match_prob, y);
      });
    }
  });
  constrain::<u16>(
    &mut thread_pool,
    &mut train_data,
    output_file_path,
    enables_randinit,
    learning_tolerance,
  );
}
