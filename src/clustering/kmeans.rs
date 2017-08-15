// http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf

use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use std::cmp::Ordering;
use std::usize;
use point::Point;
use clustering::kmeans::KMeansInitialization::*;
use statistics::distance::{Distance, SquaredEuclidean};
use statistics::statistics::Statistics;
use rayon::prelude::*;
use std::collections::HashMap;
use itertools::Itertools;

#[derive(Copy, Clone, Debug)]
pub enum KMeansInitialization {
    Random,
    KMeansPlusPlus,
    Precomputed
}

pub struct KMeans {
    assignments: Vec<usize>,
    centroids: Vec<Point>,
    iterations: usize,
    converged: bool,
    init_method: KMeansInitialization,
    precomputed: Option<Vec<Vec<f64>>>,
    max_iterations: usize,
    tolerance: f64
}

impl Default for KMeans {
    fn default() -> KMeans {
        KMeans {
            assignments: vec![],
            centroids: vec![],
            iterations: 0,
            converged: false,
            init_method: Random,
            precomputed: None,
            max_iterations: 15,
            tolerance: 0.00001
        }
    }
}

impl KMeans {
    pub fn new() -> Self {
        KMeans::default()
    }

    pub fn run(self, points: &[Point], no_clusters: usize) -> Self {
        let mut centroids = self.initial_centroids(points, no_clusters);

        let mut i = 0;
        let stop_condition = self.tolerance * self.tolerance;

        while i < self.max_iterations {
            let updated_centroids: Vec<Vec<f64>> =
                points.par_iter().fold(|| HashMap::with_capacity(no_clusters), |mut new_centroids, point| {
                    let (index_c, _) = Self::closest_centroid(point.coordinates(), centroids.as_slice());
                    (*new_centroids.entry(index_c).or_insert(vec![])).push(point.coordinates());
                    new_centroids
                }).reduce(|| HashMap::with_capacity(no_clusters), |mut new_centroids, partial| {
                    for (k, v) in partial.into_iter() {
                        (*new_centroids.entry(k).or_insert(vec![])).extend(v);
                    }

                    new_centroids
                }).into_iter().sorted_by(|a, b| a.0.cmp(&b.0)).into_iter().map(|(_, ref v)| {
                    Statistics::mean(&v)
                }).collect();

            let change = Statistics::max_change(centroids.as_slice(), updated_centroids.as_slice());
            centroids = updated_centroids;
            if change <= stop_condition {
                break;
            }

            i += 1;
        }

        KMeans {
            assignments: points.iter().map(|p| Self::closest_centroid(p.coordinates(), centroids.as_slice()).0).collect(),
            centroids: centroids.into_iter().map(|c| Point::new(c)).collect(),
            iterations: i,
            converged: i < self.max_iterations,
            .. self
        }
    }

    pub fn initial_centroids(&self, points: &[Point], no_clusters: usize) -> Vec<Vec<f64>> {
        match self.init_method {
            Random => {
                let mut rng = rand::thread_rng();
                let between = Range::new(0, points.len());

                (0..no_clusters).map(|_| {
                    points[between.ind_sample(&mut rng)].coordinates().to_vec()
                }).collect()
            },
            KMeansPlusPlus => {
                let mut rng = rand::thread_rng();
                let between = Range::new(0, points.len());

                let mut distances: Vec<f64> = vec![0.0; points.len()];
                let mut centroids: Vec<Vec<f64>> = vec![points[between.ind_sample(&mut rng)].coordinates().to_vec()];

                for _ in 1..no_clusters {
                    let mut sum = points.iter().enumerate().fold(0.0, |sum, (index_p, p)| {
                        let (_, distance_c) = Self::closest_centroid(p.coordinates(), centroids.as_slice());
                        distances[index_p] = distance_c;
                        sum + distance_c
                    });

                    sum *= rng.next_f64();
                    for (index_p, d) in distances.iter().enumerate() {
                        sum -= *d;

                        if sum <= 0f64 {
                            centroids.push(points[index_p].coordinates().to_vec());
                            break;
                        }
                    }
                }

                centroids
            },
            Precomputed => {
                vec![]//self.precomputed.expect("Expected a vec of clusters, on the form Vec<f64>").clone()
            }
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

    pub fn assignments(&self) -> &[usize] { &self.assignments }

    pub fn centroids(&self) -> &[Point] {
        &self.centroids
    }

    pub fn converged(&self) -> bool { self.converged }

    pub fn iterations(&self) -> usize { self.iterations }

    pub fn max_iterations(&self) -> usize { self.max_iterations }

    pub fn set_tolerance(self, tolerance: f64) -> Self {
        KMeans { tolerance, .. self }
    }

    pub fn set_max_iterations(self, max_iterations: usize) -> Self {
        KMeans { max_iterations, .. self }
    }

    pub fn set_init_method(self, init_method: KMeansInitialization) -> Self {
        KMeans { init_method, .. self }
    }

    pub fn set_precomputed(self, precomputed: &Option<Vec<Vec<f64>>>) -> Self {
        KMeans { precomputed: precomputed.clone(), .. self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand;
    use rand::Rng;
    use datasets::*;

    #[test]
    fn can_run_kmeans() {
        let dimension = 2;

        let mut rng = rand::thread_rng();
        let mut points: Vec<Point> = (0..10000).map(|_| {
            Point::new((0..dimension).into_iter().map(|_| rng.next_f64()).collect())
        }).collect();

        let output = KMeans::new().run(points.as_mut_slice(), 10);

        assert_eq!(points.len(), output.assignments().len());
        if output.iterations() < output.max_iterations() {
            assert!(output.converged());
        } else if output.iterations() == output.max_iterations() {
            assert!(!output.converged());
        } else {
            panic!("Algorithm should not run for more than max_iterations");
        }

        assert!(output.assignments().iter().all(|a| *a < output.centroids().len()));
        assert!(output.centroids().iter().all(|c| c.coordinates().len() == dimension));
    }

    #[test]
    fn can_run_kmeans_iris() {
        let output = KMeans::new().run(iris::load().data(), 3);

        println!("{:?}", iris::load().target());
        println!("{:?}", output.assignments());

        assert_eq!(3, output.centroids().len());
        assert!(iris::load().target().iter().zip(output.assignments().iter()).all(|(a, b)| a == b));
    }
}