extern crate consprob;

use consprob::*;
use std::env;

fn main() {
  let args = env::args().collect::<Args>();
  let program_name = args[0].clone();
  let mut opts = Options::new();
  opts.reqopt(
    "i",
    "input_dir_seq_align",
    "A path to an input directory containing RNA sequence alignments in FASTA format",
    "STR",
  );
  opts.reqopt(
    "j",
    "input_dir_second_struct",
    "A path to an input directory containing RNA secondary structures in FASTA format",
    "STR",
  );
  opts.reqopt(
    "o",
    "output_file_path_seq_align",
    "A path to an output file to record intermediate training log likelihoods in RNA sequence alignments",
    "STR",
  );
  opts.reqopt(
    "p",
    "output_file_path_second_struct",
    "A path to an output file to record intermediate training log likelihoods in RNA secondary structures",
    "STR",
  );
  opts.optopt("t", "num_of_threads", "The number of threads in multithreading (Uses the number of the threads of this computer by default)", "UINT");
  opts.optflag("h", "help", "Print a help menu");
  let matches = match opts.parse(&args[1..]) {
    Ok(opt) => opt,
    Err(failure) => {
      print_program_usage(&program_name, &opts);
      panic!(failure.to_string())
    }
  };
  if matches.opt_present("h") {
    print_program_usage(&program_name, &opts);
    return;
  }
  let input_dir_path_sa = matches.opt_str("i").unwrap();
  let input_dir_path_sa = Path::new(&input_dir_path_sa);
  let input_dir_path_ss = matches.opt_str("j").unwrap();
  let input_dir_path_ss = Path::new(&input_dir_path_ss);
  let num_of_threads = if matches.opt_present("t") {
    matches.opt_str("t").unwrap().parse().unwrap()
  } else {
    num_cpus::get() as NumOfThreads
  };
  println!("# threads = {}", num_of_threads);
  let output_file_path_sa = matches.opt_str("o").unwrap();
  let output_file_path_sa = Path::new(&output_file_path_sa);
  let output_file_path_ss = matches.opt_str("p").unwrap();
  let output_file_path_ss = Path::new(&output_file_path_ss);
  let entries: Vec<DirEntry> = read_dir(input_dir_path_sa).unwrap().map(|x| x.unwrap()).collect();
  let num_of_entries = entries.len();
  let mut train_data_sa = vec![TrainDatumSa::origin(); num_of_entries];
  for (input_file_path, train_datum) in entries.iter().zip(train_data_sa.iter_mut()) {
    *train_datum = TrainDatumSa::new(&input_file_path.path());
  }
  let entries: Vec<DirEntry> = read_dir(input_dir_path_ss).unwrap().map(|x| x.unwrap()).collect();
  let num_of_entries = entries.len();
  let mut train_data_ss = vec![TrainDatumSs::origin(); num_of_entries];
  for (input_file_path, train_datum) in entries.iter().zip(train_data_ss.iter_mut()) {
    *train_datum = TrainDatumSs::new(&input_file_path.path());
  }
  let mut thread_pool = Pool::new(num_of_threads);
  constrain::<u16>(&mut thread_pool, &mut train_data_sa, &mut train_data_ss, output_file_path_sa, output_file_path_ss);
}
