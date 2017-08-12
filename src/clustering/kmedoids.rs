use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use std::cmp::Ordering;
use std::usize;
use point::Point;
use std::collections::HashMap;
use statistics::distance::{Distance, Manhattan};
use statistics::statistics::Statistics;
use clustering::kmedoids::KMedoidsInitialization::*;
use rayon::prelude::*;

pub enum KMedoidsInitialization {
    Random,
    KMeansPlusPlus,
    Precomputed
}

pub struct KMedoids {
    assignments: Vec<usize>,
    centroids: Vec<Point>,
    iterations: usize,
    converged: bool
}

impl KMedoids {
    pub fn run(points: &[Point], no_clusters: usize, max_iterations: usize, tolerance: f64, init_method: KMedoidsInitialization) -> Self {
        let mut medoids = Self::initial_medoids(points, no_clusters, init_method);
        let mut cached_medoids: Vec<&[f64]> = medoids.iter().map(|(index_m, _)| points[*index_m].coordinates()).collect();

        let mut i = 0;
        let stop_condition = tolerance * tolerance;

        while i < max_iterations {
            let updated_medoids: HashMap<usize, Vec<usize>> =
                points.par_iter().enumerate().fold(|| HashMap::with_capacity(no_clusters), |mut new_medoids, (index_p, point)| {
                    let (index_c, _) = Self::closest_medoid(point.coordinates(), cached_medoids.as_slice());
                    (*new_medoids.entry(index_c).or_insert(vec![])).push(index_p);
                    new_medoids
                }).reduce(|| HashMap::with_capacity(no_clusters), |mut new_medoids, partial| {
                    new_medoids.extend(partial);
                    new_medoids
                }).into_iter().map(|(index_c, cluster_points)| {
                    let current_cost = cluster_points.iter().map(|index_p| {
                        Manhattan::distance(points[*index_p].coordinates(), points[index_c].coordinates())
                    }).sum();

                    let (medoid, _) = match cluster_points.iter().map(|candidate_medoid| {
                        let old = Manhattan::distance(points[index_c].coordinates(), points[*candidate_medoid].coordinates());
                        let new = cluster_points
                                    .iter()
                                    .filter(|index_p| *index_p != candidate_medoid)
                                    .map(|index_p| {
                                        Manhattan::distance(points[*index_p].coordinates(), points[*candidate_medoid].coordinates())
                                    })
                                    .sum::<f64>();

                        (candidate_medoid, old + new)
                    }).min_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap_or(Ordering::Equal)) {
                        Some((candidate, distance)) if distance < current_cost => (*candidate, distance),
                        Some(_) => (index_c, current_cost),
                        None => panic!()
                    };

                    (medoid, vec![])
                }).collect();

            let updated_centroids: Vec<&[f64]> = updated_medoids.iter().map(|(index_m, _)| points[*index_m].coordinates()).collect();
            let change = Statistics::max_change_slice(cached_medoids.as_slice(), updated_centroids.as_slice());
            medoids = updated_medoids;
            cached_medoids = updated_centroids;
            if change <= stop_condition {
                break;
            }

            i += 1;
        }

        KMedoids {
            assignments: points.iter().map(|p| Self::closest_medoid(p.coordinates(), cached_medoids.as_slice()).0).collect(),
            centroids: medoids.into_iter().map(|(index_m, _)| points[index_m].clone()).collect(),
            iterations: i,
            converged: i < max_iterations
        }
    }

    fn initial_medoids(points: &[Point], no_clusters: usize, init_method: KMedoidsInitialization) -> HashMap<usize, Vec<usize>> {
        match init_method {
            Random => {
                let mut rng = rand::thread_rng();
                let between = Range::new(0, points.len());

                (0..no_clusters).map(|_| {
                    (between.ind_sample(&mut rng), vec![])
                }).collect()
            },
            KMeansPlusPlus => {
                let mut rng = rand::thread_rng();
                let between = Range::new(0, points.len());

                let mut distances: Vec<f64> = vec![0.0; points.len()];
                let mut centroids: Vec<&[f64]> = vec![points[between.ind_sample(&mut rng)].coordinates()];
                let mut result: HashMap<usize, Vec<usize>> = HashMap::with_capacity(no_clusters);

                for _ in 1..no_clusters {
                    let mut sum = points.iter().enumerate().fold(0.0, |sum, (index_p, p)| {
                        let (_, distance_c) = Self::closest_medoid(p.coordinates(), centroids.as_slice());
                        distances[index_p] = distance_c;
                        sum + distance_c
                    });

                    sum *= rng.next_f64();
                    for (index_p, d) in distances.iter().enumerate() {
                        sum -= *d;

                        if sum <= 0f64 {
                            centroids.push(points[index_p].coordinates());
                            result.insert(index_p, vec![]);
                            break;
                        }
                    }
                }

                result
            },
            Precomputed => {
                HashMap::new()
            }
        }
    }

    #[inline]
    fn closest_medoid(point: &[f64], centroids: &[&[f64]]) -> (usize, f64) {
        match centroids.iter().enumerate().map(|(index_c, c)| {
            (index_c, Manhattan::distance(point, c))
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
    fn bench_10000_points_kmedoids() {
        let mut rng = rand::thread_rng();
        let mut points: Vec<Point> = (0..100000).map(|_| {
            Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
        }).collect();

        let repeat_count = 10_u8;
        let mut total = 0_u64;
        for _ in 0..repeat_count {
            let start = time::precise_time_ns();
            KMedoids::run(points.as_mut_slice(), 10, 15, 0.00001, KMedoidsInitialization::Random);
            let end = time::precise_time_ns();
            total += end - start
        }

        let avg_ns: f64 = total as f64 / repeat_count as f64;
        let avg_ms = avg_ns / 1.0e6;

        println!("{} runs, avg {}", repeat_count, avg_ms);
    }
}