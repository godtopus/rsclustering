// https://www.cis.upenn.edu/~sudipto/mypapers/cure_final.pdf

use std::collections::BinaryHeap;
use cluster::Cluster;
use point::Point;
use std::fmt::Debug;

pub fn cure<T: Clone + Copy + Debug>(points: Vec<Point<T>>, no_clusters: usize, number_represent_points: usize, compression: f64) -> Vec<Cluster<T>> {
    let mut clusters = points.into_iter().map(|p| Cluster::new(p)).collect::<BinaryHeap<Cluster<T>>>();

    while clusters.len() > no_clusters {
        let merged = match clusters.pop() {
            Some(cluster) => cluster.merge(number_represent_points, compression),
            None => break
        };

        match merged {
            Some(cluster) => {
                clusters = clusters.into_iter().map(|mut c| {
                    c.update_closest();
                    c
                }).collect();

                clusters.push(cluster);
            },
            None => panic!()
        }
    }

    clusters.into_vec()
}