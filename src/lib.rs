//#![feature(test)]
#![allow(dead_code)]
#![allow(unused_imports)]

extern crate rand;
extern crate time;
extern crate itertools;
extern crate rayon;
extern crate nalgebra;

pub mod clustering {
    pub mod agglomerative;
    pub mod fuzzy_c_means;
    pub mod kmeans;
    pub mod kmedians;
    pub mod kmedoids;
    pub mod mini_batch_kmeans;
}

pub mod statistics {
    pub mod statistics;
    pub mod distance;
}

pub mod experimental {
    pub mod kdtree;
    pub mod cure;
    pub mod xmeans;
    pub mod clarans;
    pub mod cluster;
}

pub mod point;