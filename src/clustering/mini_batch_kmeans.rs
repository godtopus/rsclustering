use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use std::cmp::Ordering;
use std::usize;
use point::Point;
use statistics::distance::{Distance, SquaredEuclidean};
use statistics::statistics::Statistics;
use clustering::kmeans::*;

pub struct MiniBatchKMeans {
    assignments: Vec<usize>,
    centroids: Vec<Point>,
    iterations: usize,
    converged: bool,
    init_method: KMeansInitialization,
    precomputed: Option<Vec<Vec<f64>>>,
    max_iterations: usize,
    tolerance: f64,
    batch_size: usize
}

impl Default for MiniBatchKMeans {
    fn default() -> MiniBatchKMeans {
        MiniBatchKMeans {
            assignments: vec![],
            centroids: vec![],
            iterations: 0,
            converged: false,
            init_method: KMeansInitialization::Random,
            precomputed: None,
            max_iterations: 15,
            tolerance: 0.00001,
            batch_size: 10000
        }
    }
}

impl MiniBatchKMeans {
    pub fn new() -> Self {
        MiniBatchKMeans::default()
    }

    pub fn run(self, points: &[Point], no_clusters: usize) -> Self {
        let mut centroids = KMeans::new().set_init_method(self.init_method).set_precomputed(&self.precomputed).initial_centroids(points, no_clusters);
        let mut cluster_size = vec![0.0; no_clusters];

        let mut rng = rand::thread_rng();
        let between = Range::new(0, points.len());

        let mut i = 0;
        let stop_condition = self.tolerance * self.tolerance;

        while i < self.max_iterations {
            let previous_centroids = centroids.clone();

            for _ in 0..self.batch_size {
                let p = points[between.ind_sample(&mut rng)].coordinates();

                let (index_c, _) =  Self::closest_centroid(p, centroids.as_slice());
                cluster_size[index_c] += 1.0;

                // Gradient descent
                let eta = 1.0 / cluster_size[index_c];
                let eta_compliment = 1.0 - eta;
                centroids[index_c] = centroids[index_c].iter().zip(p.iter()).map(|(c, p)| {
                    eta_compliment * c + eta * p
                }).collect();
            };

            let change = Statistics::max_change(previous_centroids.as_slice(), centroids.as_slice());
            if change < stop_condition {
                break;
            }

            i += 1;
        }

        MiniBatchKMeans {
            assignments: vec![],
            centroids: centroids.into_iter().map(|c| Point::new(c)).collect(),
            iterations: i,
            converged: i < self.max_iterations,
            .. self
        }
    }

    #[inline]
    fn closest_centroid(point: &[f64], centroids: &[Vec<f64>]) -> (usize, f64) {
        match centroids.iter().enumerate().map(|(index_c, c)| {
            (index_c, SquaredEuclidean::distance(point, c))
        }).min_by(|&(_, a), &(_, b)| {
            a.partial_cmp(&b).unwrap_or(Ordering::Equal)
        }) {
            Some(closest) => closest,
            None => panic!()
        }
    }

    pub fn centroids(&self) -> &[Point] {
        &self.centroids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand;
    use rand::Rng;
    use time;

    /*#[test]
    fn can_run() {
        let expected: Vec<KMeansCluster> = vec![
            KMeansCluster::from(vec![Point::new(vec![0.0, 1.0]), Point::new(vec![1.0, 2.0]), Point::new(vec![2.0, 3.0])]),
            KMeansCluster::from(vec![Point::new(vec![3.0, 4.0]), Point::new(vec![4.0, 5.0])])
        ];
        let mut input = vec![Point::new(vec![0.0, 1.0]), Point::new(vec![1.0, 2.0]), Point::new(vec![2.0, 3.0]), Point::new(vec![3.0, 4.0]), Point::new(vec![4.0, 5.0])];

        let output = kmeans(input.as_mut_slice(), 2, usize::max_value());

        assert_eq!(expected, output);
    }*/
}