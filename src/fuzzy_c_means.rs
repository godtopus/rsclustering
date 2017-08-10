use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use std::cmp::Ordering;
use std::usize;
use std::f64;
use point::Point;
use distance::*;
use fuzzy_c_means::FuzzyCMeansInitialization::*;
use statistics::Statistics;
use rayon::prelude::*;

use std::sync::{Mutex};

pub enum FuzzyCMeansInitialization {
    Random,
    KMeansPlusPlus,
    Precomputed
}

pub struct FuzzyCMeans {
    assignments: Vec<usize>,
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

        let mut previous_round: Vec<Vec<f64>> = vec![vec![1.0 / no_clusters as f64; no_clusters]; points.len()];

        let mut i = 0;

        while i < max_iterations {
            let max_delta = Mutex::new(f64::INFINITY);

            previous_round = points.par_iter().zip(previous_round.par_iter()).map(|(p, previous_memberships)| {
                let memberships = Self::memberships(p.coordinates(), centroids.as_slice(), fuzziness);

                let delta = SquaredEuclidean::distance(&memberships, previous_memberships);
                let mut max_delta = max_delta.lock().unwrap();
                if delta < *max_delta {
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
            assignments: vec![],//previous_round.into_iter().sorted_by(|&(p1, _), &(p2, _)| p1.partial_cmp(&p2).unwrap_or(Ordering::Equal)).into_iter().map(|(_, c)| c).collect(),
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
            KMeansPlusPlus => {
                /*let mut rng = rand::thread_rng();
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

                centroids*/
                vec![]
            },
            Precomputed => {
                precomputed.expect("Expected a slice of clusters, on the form Vec<f64>").to_vec()
            }
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
    //use test::Bencher;

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

    #[test]
    fn bench_100000_points_fuzzycmeans() {
        let mut rng = rand::thread_rng();
        let mut points: Vec<Point> = (0..100000).map(|_| {
            Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
        }).collect();

        let repeat_count = 10_u8;
        let mut total = 0_u64;
        for _ in 0..repeat_count {
            let start = time::precise_time_ns();
            FuzzyCMeans::run(points.as_mut_slice(), 10, 15, 1.5, 0.01, FuzzyCMeansInitialization::Random, None);
            let end = time::precise_time_ns();
            total += end - start
        }

        let avg_ns: f64 = total as f64 / repeat_count as f64;
        let avg_ms = avg_ns / 1.0e6;

        println!("{} runs, avg {}", repeat_count, avg_ms);
    }

    /*#[bench]
    fn bench_100000_points(b: &mut Bencher) {
        let mut rng = rand::thread_rng();
        let points = Vec::from_fn(100000, |_| Point::new(vec![rng.next_f64(), rng.next_f64()]));

        b.iter(|| {
            kmeans(points, 50, usize::max_value());
        });
    }*/
}