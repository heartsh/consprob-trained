# Trainable Probability Inference Engine on RNA Structural Alignment
# Installation
This project is written in Rust, a systems programming language.
You need to install Rust components, i.e., rustc (the Rust compiler), cargo (the Rust package manager), and the Rust standard library.
Visit [the Rust website](https://www.rust-lang.org) to see more about Rust.
You can install Rust components with the following one line:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
[Rustup](https://github.com/rust-lang-nursery/rustup.rs) arranges the above installation and enables to switch a compiler in use easily.
You can install ConsProb:
```bash
# AVX, SSE, and MMX enabled for rustc
# Another example: RUSTFLAGS='--emit asm -C target-feature=+avx2 -C target-feature=+ssse3 -C target-feature=+mmx -C target-feature=+fma'
RUSTFLAGS='--emit asm -C target-feature=+avx -C target-feature=+ssse3 -C target-feature=+mmx' \
  cargo install consprob-trained
```
Check if you have installed ConsProb properly:
```bash
# Its available command options will be displayed
consprob_trained
```
You can run ConsProb with a prepared test set of sampled tRNAs:
```bash
git clone https://github.com/heartsh/consprob-trained \
  && cd consprob-trained
cargo test --release
# The below command requires Gnuplot (http://www.gnuplot.info)
# Benchmark results will be found at "./target/criterion/report/index.html"
cargo bench
```

# Docker Playground <img src="./assets/images_fixed/docker_logo.png" width="40">
I offer [my Docker-based playground for RNA software and its instruction](https://github.com/heartsh/rna-playground) to replay my computational experiments easily.

# Method Digest
[ConsProb-Turner](https://github.com/heartsh/consprob) can compute a variety of sparse posterior probabilities on RNA pairwise structural alignment using [Turner's model](https://github.com/heartsh/rna-ss-params).
This repository offers ConsProb-trained, a machine-learning counterpart of ConsProb-Turner.
This repository also includes a ConsTrain, structural alignment-based machine-learning method for ConsProb-trained.

# Author
[Heartsh](https://github.com/heartsh)

# License
Copyright (c) 2018 Heartsh  
Licensed under [the MIT license](http://opensource.org/licenses/MIT).
