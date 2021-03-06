use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use std::cmp::Ordering;
use std::usize;
use point::Point;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use experimental::xmeans::XMeansInitialization::*;
use statistics::distance::{Distance, SquaredEuclidean};
use clustering::kmeans::*;

pub struct XMeansDefaults {
    sample_percent: Option<f64>,
    learning_rate: Option<f64>
}

pub enum XMeansInitialization {
    Random,
    KMeansPlusPlus,
    Precomputed
}

pub enum XMeansSplittingCriterion {
    BIC,
    MNDL
}

pub struct XMeans {
    assignments: Vec<usize>,
    centroids: Vec<Point>,
    iterations: usize,
    converged: bool
}

impl XMeans {
    pub fn run(points: &[Point], no_clusters_min: usize, no_clusters_max: Option<usize>, init_method: XMeansInitialization, precomputed: Option<&[Vec<f64>]>) -> Self {
        let mut centroids = Self::initial_centroids(points, no_clusters_min, init_method, precomputed);

        let mut k = no_clusters_min;

        while no_clusters_max == None || k <= no_clusters_max.unwrap() {
            // 1. Improve-Params
            let kmeans = KMeans::new();// KMeans::run(points, k, 100, 0.00001, KMeansInitialization::Precomputed, Some(&centroids));
            let model: Vec<Vec<f64>> = kmeans.centroids().iter().map(|c| c.coordinates().to_vec()).collect();
            let centroid_distances: Vec<Vec<f64>> = model.iter().map(|m| model.iter().map(|other_m| SquaredEuclidean::distance(m, other_m)).collect()).collect();

            for centroid in model.iter() {

            }

            if model.len() == centroids.len() {
                break;
            } else {
                centroids = model;
            }

            k += 1;
        }

        XMeans {
            assignments: vec![], //previous_round.into_iter().sorted_by(|&(p1, _), &(p2, _)| p1.partial_cmp(&p2).unwrap_or(Ordering::Equal)).into_iter().map(|(_, c)| c).collect(),
            centroids: centroids.into_iter().map(|c| Point::new(c)).collect(),
            iterations: k - no_clusters_min,
            converged: true // TODO
        }
    }

    fn initial_centroids(points: &[Point], no_clusters: usize, init_method: XMeansInitialization, precomputed: Option<&[Vec<f64>]>) -> Vec<Vec<f64>> {
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

    //#[test]
    fn bench_100000_points() {
        let mut rng = rand::thread_rng();
        let mut points: Vec<Point> = (0..100000).map(|_| {
            Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
        }).collect();

        let repeat_count = 10_u8;
        let mut total = 0_u64;
        for _ in 0..repeat_count {
            let start = time::precise_time_ns();
            XMeans::run(points.as_mut_slice(), 2, Some(10), XMeansInitialization::Random, None);
            let end = time::precise_time_ns();
            total += end - start
        }

        let avg_ns: f64 = total as f64 / repeat_count as f64;
        let avg_ms = avg_ns / 1.0e6;

        println!("{} runs, avg {}", repeat_count, avg_ms);
    }
}