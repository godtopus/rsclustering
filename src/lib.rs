//#![feature(test)]

pub mod cure;
pub mod kmeans;
pub mod kmedians;
pub mod kmedoids;
pub mod xmeans;
pub mod agglomerative;

pub mod kmeanscluster;
pub mod cluster;

pub mod distance;
pub mod point;
pub mod statistics;

pub mod kdtree;

extern crate rand;
extern crate time;
extern crate itertools;
extern crate rayon;