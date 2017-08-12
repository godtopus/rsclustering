use rand;
use rand::distributions::{IndependentSample, Range};

use std::cmp::Ordering;
use std::usize;
use std::f64;
use point::Point;
use distance::*;
use std::collections::HashMap;
use std::collections::HashSet;
use rayon::prelude::*;

pub struct Clarans {
    assignments: Vec<usize>,
    centroids: Vec<Point>,
    iterations: usize,
    converged: bool
}

impl Clarans {
    pub fn run(points: &[Point], no_clusters: usize, num_local: usize, max_neighbor: usize) -> Self {
        let mut optimal_medoids = vec![];
        let mut optimal_estimation = f64::INFINITY;

        let mut rng = rand::thread_rng();
        let point_range = Range::new(0, points.len());
        let medoid_range = Range::new(0, no_clusters);

        for _ in 0..num_local {
            let mut current_indexes = HashSet::with_capacity(no_clusters);

            let mut medoids: Vec<(usize, &[f64])> = (0..points.len()).map(|_| {
                let index = point_range.ind_sample(&mut rng);
                current_indexes.insert(index);
                (index, points[index].coordinates())
            }).collect();

            let mut assignments: HashMap<usize, usize> = points.iter().enumerate().map(|(index_p, p)| {
                (index_p, Self::closest_centroid(p.coordinates(), medoids.as_slice()).0)
            }).collect();

            let mut index_neighbor = 0;
            while index_neighbor < max_neighbor {
                let current_index = medoid_range.ind_sample(&mut rng);
                let (current_medoid_index, current_medoid_coordinates) = medoids[current_index];
                let current_medoid_cluster_index = *assignments.get(&current_medoid_index).unwrap();

                let mut candidate_medoid_index = point_range.ind_sample(&mut rng);

                while current_indexes.contains(&candidate_medoid_index) {
                    candidate_medoid_index = point_range.ind_sample(&mut rng);
                }

                let candidate_cost: f64 = points.par_iter().enumerate().filter(|&(index_p, _)| !current_indexes.contains(&index_p)).map(|(index_p, p)| {
                    let point_cluster_index = *assignments.get(&index_p).unwrap();
                    let (point_medoid_index, point_medoid_coordinates) = medoids[point_cluster_index];

                    let (_, other_medoid_index, _) = Self::closest_centroid_not_in(p.coordinates(), medoids.as_slice(), current_medoid_index);
                    let other_medoid_cluster_index = *assignments.get(&other_medoid_index).unwrap();

                    let distance_current = SquaredEuclidean::distance(p.coordinates(), current_medoid_coordinates);
                    let distance_candidate = SquaredEuclidean::distance(p.coordinates(), points[candidate_medoid_index].coordinates());

                    let distance_nearest = match (point_medoid_index != candidate_medoid_index) && (point_medoid_index != current_medoid_cluster_index) {
                        true => SquaredEuclidean::distance(p.coordinates(), point_medoid_coordinates),
                        false => f64::INFINITY
                    };

                    match point_cluster_index {
                        i if i == current_medoid_cluster_index => {
                            if distance_candidate >= distance_nearest {
                                distance_nearest - distance_current
                            } else {
                                distance_candidate - distance_current
                            }
                        },
                        i if i == other_medoid_cluster_index => {
                            if distance_candidate <= distance_nearest {
                                distance_candidate - distance_nearest
                            } else {
                                0.0
                            }
                        },
                        _ => 0.0
                    }
                }).sum();

                if candidate_cost < -1.0 {
                    medoids[current_index] = (candidate_medoid_index, points[candidate_medoid_index].coordinates());
                    assignments = points.iter().enumerate().map(|(index_p, p)| {
                        (index_p, Self::closest_centroid(p.coordinates(), medoids.as_slice()).0)
                    }).collect();

                    current_indexes.remove(&current_medoid_index);
                    current_indexes.insert(candidate_medoid_index);

                    index_neighbor = 0;
                } else {
                    index_neighbor += 1;
                }
            }

            let estimation = points.iter().map(|p| Self::closest_centroid(p.coordinates(), medoids.as_slice()).2).sum();
            if estimation < optimal_estimation {
                optimal_medoids = medoids;
                optimal_estimation = estimation;
            }
        }

        Clarans {
            assignments: points.iter().map(|p| Self::closest_centroid(p.coordinates(), optimal_medoids.as_slice()).0).collect(),
            centroids: optimal_medoids.into_iter().map(|(index_m, _)| points[index_m].clone()).collect(),
            iterations: 0,
            converged: true
        }
    }

    #[inline]
    fn closest_centroid(point: &[f64], centroids: &[(usize, &[f64])]) -> (usize, usize, f64) {
        Self::closest_centroid_not_in(point, centroids, usize::max_value())
    }

    #[inline]
    fn closest_centroid_not_in(point: &[f64], centroids: &[(usize, &[f64])], not_in: usize) -> (usize, usize, f64) {
        match centroids.iter().enumerate().filter(|&(index_m, _)| index_m != not_in).map(|(index_m, &(index_c, c))| {
            (index_m, index_c, SquaredEuclidean::distance(point, c))
        }).min_by(|&(_, _, a), &(_, _, b)| {
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
    fn bench_100000_points_clarans() {
        let mut rng = rand::thread_rng();
        let mut points: Vec<Point> = (0..1000).map(|_| {
            Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
        }).collect();

        let repeat_count = 1_u8;
        let mut total = 0_u64;
        for _ in 0..repeat_count {
            let start = time::precise_time_ns();
            Clarans::run(points.as_mut_slice(), 10, 10, 10);
            let end = time::precise_time_ns();
            total += end - start
        }

        let avg_ns: f64 = total as f64 / repeat_count as f64;
        let avg_ms = avg_ns / 1.0e6;

        println!("{} runs, avg {}", repeat_count, avg_ms);
    }
}