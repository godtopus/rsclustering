// http://cse.ucdenver.edu/~cscialtman/CVData/P_2011.pdf
// https://www.researchgate.net/publication/3335850_Reducing_the_time_complexity_of_the_fuzzy_c-means_algorithm

use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use std::cmp::Ordering;
use std::usize;
use std::f64;
use point::Point;
use clustering::fuzzy_cmeans::FuzzyCMeansInitialization::*;
use statistics::distance::{Distance, SquaredEuclidean};
use statistics::statistics::Statistics;
use rayon::prelude::*;

use std::sync::{Mutex};

#[derive(Copy, Clone, Debug)]
pub enum FuzzyCMeansInitialization {
    Random,
    FuzzyCMeansPlusPlus,
    Precomputed
}

pub struct FuzzyCMeans {
    assignments: Vec<Vec<f64>>,
    centroids: Vec<Point>,
    iterations: usize,
    converged: bool
}

impl FuzzyCMeans {
    pub fn run(points: &[Point], no_clusters: usize, max_iterations: usize, fuzziness: f64, epsilon: f64, init_method: FuzzyCMeansInitialization, precomputed: Option<&[Vec<f64>]>) -> Self {
        if fuzziness <= 1.0 {
            panic!()
        }

        if epsilon <= 0.0 || epsilon > 1.0 {
            panic!()
        }

        let dimension = points[0].coordinates().len();

        let mut centroids = Self::initial_centroids(points, no_clusters, init_method, precomputed);

        let mut previous_round: Vec<Vec<f64>> = points.par_iter().map(|p| Self::memberships(p.coordinates(), &centroids, fuzziness)).collect();

        let mut i = 0;

        while i < max_iterations {
            let max_delta = Mutex::new(f64::NEG_INFINITY);

            previous_round = points.par_iter().zip(previous_round.par_iter()).map(|(p, previous_memberships)| {
                let memberships = Self::memberships(p.coordinates(), centroids.as_slice(), fuzziness);

                let delta = SquaredEuclidean::distance(&memberships, previous_memberships);
                let mut max_delta = max_delta.lock().unwrap();
                if delta > *max_delta {
                    *max_delta = delta;
                }

                memberships
            }).collect();

            if *max_delta.lock().unwrap() < epsilon {
                break;
            }

            centroids = previous_round.iter().zip(points.iter()).fold(vec![vec![(0.0, 0.0); dimension]; no_clusters], |mut clusters, (memberships, point)| {
                let coordinates = point.coordinates();

                for i in 0..clusters.len() {
                    let membership = memberships[i].powf(fuzziness);

                    for j in 0..dimension {
                        clusters[i][j] = (clusters[i][j].0 + (membership * coordinates[j]), clusters[i][j].1 + membership);
                    }
                }

                clusters
            })
            .into_iter()
            .map(|cluster| cluster.iter().map(|&(numerator, denominator)| numerator / denominator).collect())
            .collect();

            i += 1;
        }

        FuzzyCMeans {
            assignments: previous_round,
            centroids: centroids.into_iter().map(|c| Point::new(c)).collect(),
            iterations: i,
            converged: i < max_iterations
        }
    }

    fn initial_centroids(points: &[Point], no_clusters: usize, init_method: FuzzyCMeansInitialization, precomputed: Option<&[Vec<f64>]>) -> Vec<Vec<f64>> {
        match init_method {
            Random => {
                let mut rng = rand::thread_rng();
                let between = Range::new(0, points.len());

                (0..no_clusters).map(|_| {
                    points[between.ind_sample(&mut rng)].coordinates().to_vec()
                }).collect()
            },
            FuzzyCMeansPlusPlus => {
                let mut rng = rand::thread_rng();
                let between = Range::new(0, points.len());

                let mut centroids: Vec<Vec<f64>> = vec![points[between.ind_sample(&mut rng)].coordinates().to_vec()];

                for _ in 1..no_clusters {
                    let mut sum = 0.0;
                    let distances: Vec<(usize, f64)> = points.iter().enumerate().map(|(index_p, p)| {
                        let (_, distance_c) = Self::closest_centroid(p.coordinates(), centroids.as_slice());
                        sum + distance_c;
                        (index_p, distance_c)
                    }).collect();

                    sum *= rng.next_f64();
                    for (index_p, d) in distances.into_iter() {
                        sum -= d;

                        if sum <= 0f64 {
                            centroids.push(points[index_p].coordinates().to_vec());
                            break;
                        }
                    }
                }

                centroids
            },
            Precomputed => {
                precomputed.expect("Expected a slice of clusters, on the form Vec<f64>").to_vec()
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

    #[inline]
    fn memberships(point: &[f64], centroids: &[Vec<f64>], fuzziness: f64) -> Vec<f64> {
        centroids.iter().map(|c_j| {
            let distance = SquaredEuclidean::distance(point, c_j);
            let total_distance = centroids.iter().map(|c_k| {
                (distance / SquaredEuclidean::distance(point, c_k)).powf(2.0 / (fuzziness - 1.0))
            }).sum::<f64>();

            1.0 / total_distance
        }).collect()
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