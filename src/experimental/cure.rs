// https://www.cis.upenn.edu/~sudipto/mypapers/cure_final.pdf

use std::collections::BinaryHeap;
use experimental::cluster::Cluster;
use point::Point;

pub fn cure(points: Vec<Point>, no_clusters: usize, number_represent_points: usize, compression: f64) -> Vec<Cluster> {
    let mut clusters = points.into_iter().map(|p| Cluster::new(p)).collect::<BinaryHeap<Cluster>>();

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