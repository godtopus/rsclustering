use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use std::cmp::Ordering;
use std::usize;
use point::Point;
use distance::*;
use kmedians::KMediansInitialization::*;
use rayon::prelude::*;
use statistics::*;
use std::collections::HashMap;

pub enum KMediansInitialization {
    Random,
    KMeansPlusPlus,
    Precomputed
}

pub struct KMedians {
    assignments: Vec<usize>,
    centroids: Vec<Point>,
    iterations: usize,
    converged: bool
}

impl KMedians {
    pub fn run(points: &[Point], no_clusters: usize, max_iterations: usize, tolerance: f64, init_method: KMediansInitialization, precomputed: Option<&[Vec<f64>]>) -> Self {
        let dimension = points[0].coordinates().len() as f64;

        let mut centroids = Self::initial_centroids(points, no_clusters, init_method, precomputed);

        let mut i = 0;
        let stop_condition = tolerance * tolerance;

        while i < max_iterations {
            let updated_centroids: Vec<Vec<f64>> =
                points.par_iter().fold(|| HashMap::with_capacity(no_clusters), |mut new_centroids, point| {
                    let (index_c, _) = Self::closest_centroid(point.coordinates(), centroids.as_slice());
                    (*new_centroids.entry(index_c).or_insert(vec![])).push(point.coordinates());
                    new_centroids
                }).reduce(|| HashMap::with_capacity(no_clusters), |mut new_centroids, partial| {
                    new_centroids.extend(partial);
                    new_centroids
                }).into_iter().map(|(_, ref mut v)| {
                    let relative_index_median = v.len() / 2;

                    (0..dimension as usize).map(|index_dimension| {
                        v.sort_by(|p1, p2| p1[index_dimension].partial_cmp(&p2[index_dimension]).unwrap_or(Ordering::Equal));

                        if v.len() % 2 == 0 {
                            ((v[relative_index_median - 1][index_dimension] + v[relative_index_median][index_dimension]) as f64) / 2.0
                        } else {
                            v[relative_index_median][index_dimension]
                        }
                    }).collect()
                }).collect();

            let change = Statistics::max_change(centroids.as_slice(), updated_centroids.as_slice());
            centroids = updated_centroids;
            if change <= stop_condition {
                break;
            }

            i += 1;
        }

        KMedians {
            assignments: points.iter().map(|p| Self::closest_centroid(p.coordinates(), centroids.as_slice()).0).collect(),
            centroids: centroids.into_iter().map(|c| Point::new(c)).collect(),
            iterations: i,
            converged: i < max_iterations
        }
    }

    fn initial_centroids(points: &[Point], no_clusters: usize, init_method: KMediansInitialization, precomputed: Option<&[Vec<f64>]>) -> Vec<Vec<f64>> {
        match init_method {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand;
    use rand::Rng;
    use time;

    #[test]
    fn bench_100000_points_kmedians() {
        let mut rng = rand::thread_rng();
        let mut points: Vec<Point> = (0..100000).map(|_| {
            Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
        }).collect();

        let repeat_count = 10_u8;
        let mut total = 0_u64;
        for _ in 0..repeat_count {
            let start = time::precise_time_ns();
            KMedians::run(points.as_mut_slice(), 10, 15, 0.00001, KMediansInitialization::Random, None);
            let end = time::precise_time_ns();
            total += end - start
        }

        let avg_ns: f64 = total as f64 / repeat_count as f64;
        let avg_ms = avg_ns / 1.0e6;

        println!("{} runs, avg {}", repeat_count, avg_ms);
    }
}