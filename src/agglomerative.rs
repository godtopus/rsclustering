use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use std::cmp::Ordering;
use std::usize;
use std::f64;
use point::Point;
use distance::*;
use std::collections::HashMap;
use itertools::Itertools;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use agglomerative::Link::*;
use statistics::Statistics;

pub enum Link {
    Single,
    Complete,
    Average,
    Centroid
}

struct Cluster {
    points: Vec<usize>,
    centroid: Vec<f64>
}

pub struct Agglomerative {
    assignments: Vec<usize>,
    centroids: Vec<Point>,
    iterations: usize,
    converged: bool
}

impl Agglomerative {
    pub fn run(points: &[Point], no_clusters: usize, link_criterion: &Link) -> Self {
        let mut clusters = match *link_criterion {
            Single => (0..points.len()).map(|p| {
                Cluster {
                    points: vec![p],
                    centroid: vec![]
                }
            }).collect::<Vec<Cluster>>(),
            Complete => (0..points.len()).map(|p| {
                Cluster {
                    points: vec![p],
                    centroid: vec![]
                }
            }).collect::<Vec<Cluster>>(),
            Average => (0..points.len()).map(|p| {
                Cluster {
                    points: vec![p],
                    centroid: vec![]
                }
            }).collect::<Vec<Cluster>>(),
            Centroid => points.iter().enumerate().map(|(index, p)| {
                Cluster {
                    points: vec![index],
                    centroid: p.coordinates().to_vec()
                }
            }).collect::<Vec<Cluster>>(),
        };

        while clusters.len() > no_clusters {
            clusters = Self::merge_clusters(points, clusters, link_criterion);
        }

        Agglomerative {
            assignments: vec![],
            centroids: vec![],
            iterations: 1,
            converged: true
        }
    }

    fn merge_clusters(points: &[Point], clusters: Vec<Cluster>, link_criterion: &Link) -> Vec<Cluster> {
        match *link_criterion {
            Single => Self::merge_by_single_link(points, clusters),
            Complete => Self::merge_by_complete_link(points, clusters),
            Average => Self::merge_by_average_link(points, clusters),
            Centroid => Self::merge_by_centroid_link(clusters)
        }
    }

    fn merge_by_average_link(points: &[Point], mut clusters: Vec<Cluster>) -> Vec<Cluster> {
        let ((closest1, closest2), _) = match clusters.iter().enumerate().map(|(index_c1, cluster1)| {
            match clusters.iter().skip(index_c1 + 1).enumerate().map(|(index_c2, cluster2)| {
                let avg_distance = cluster1.points.iter().map(|point_c1| {
                    cluster2.points.iter().map(|point_c2| {
                        SquaredEuclidean::distance(points[*point_c1].coordinates(), points[*point_c2].coordinates())
                    }).sum::<f64>()
                }).sum::<f64>() / ((cluster1.points.len() + cluster2.points.len()) as f64);

                ((index_c1, index_c2), avg_distance)
            }).min_by(|&(_, a), &(_, b)| {
                a.partial_cmp(&b).unwrap_or(Ordering::Equal)
            }) {
                Some(closest) => closest,
                None => ((usize::max_value(), usize::max_value()), f64::INFINITY)
            }
        }).min_by(|&(_, a), &(_, b)| {
            a.partial_cmp(&b).unwrap_or(Ordering::Equal)
        }) {
            Some(closest) => closest,
            None => panic!("Expected at least two clusters, found none")
        };

        {
            let cluster2 = clusters.remove(closest2);
            let cluster1 = clusters.get_mut(closest1).unwrap();
            cluster1.points.extend(cluster2.points);
        }

        clusters
    }

    fn merge_by_centroid_link(mut clusters: Vec<Cluster>) -> Vec<Cluster> {
        let ((closest1, closest2), _) = match clusters.iter().enumerate().map(|(index_c1, cluster1)| {
            match clusters.iter().skip(index_c1 + 1).enumerate().map(|(index_c2, cluster2)| {
                ((index_c1, index_c2), SquaredEuclidean::distance(&cluster1.centroid, &cluster2.centroid))
            }).min_by(|&(_, a), &(_, b)| {
                a.partial_cmp(&b).unwrap_or(Ordering::Equal)
            }) {
                Some(closest) => closest,
                None => ((usize::max_value(), usize::max_value()), f64::INFINITY)
            }
        }).min_by(|&(_, a), &(_, b)| {
            a.partial_cmp(&b).unwrap_or(Ordering::Equal)
        }) {
            Some(closest) => closest,
            None => panic!("Expected at least two clusters, found none")
        };

        {
            let cluster2 = clusters.remove(closest2);
            let cluster1 = clusters.get_mut(closest1).unwrap();
            cluster1.points.extend(cluster2.points);
            cluster1.centroid = Statistics::mean(&[&cluster1.centroid, &cluster2.centroid]);
        }

        clusters
    }

    fn merge_by_complete_link(points: &[Point], mut clusters: Vec<Cluster>) -> Vec<Cluster> {
        let ((closest1, closest2), _) = match clusters.iter().enumerate().map(|(index_c1, cluster1)| {
            match clusters.iter().skip(index_c1 + 1).enumerate().map(|(index_c2, cluster2)| {
                let max_distance = match cluster1.points.iter().map(|point_c1| {
                    match cluster2.points.iter().map(|point_c2| {
                        SquaredEuclidean::distance(points[*point_c1].coordinates(), points[*point_c2].coordinates())
                    }).max_by(|a, b| {
                        a.partial_cmp(&b).unwrap_or(Ordering::Equal)
                    }) {
                        Some(closest) => closest,
                        None => f64::NEG_INFINITY
                    }
                }).max_by(|a, b| {
                    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
                }) {
                    Some(closest) => closest,
                    None => f64::NEG_INFINITY
                };

                ((index_c1, index_c2), max_distance)
            }).max_by(|&(_, a), &(_, b)| {
                a.partial_cmp(&b).unwrap_or(Ordering::Equal)
            }) {
                Some(closest) => closest,
                None => ((usize::max_value(), usize::max_value()), f64::NEG_INFINITY)
            }
        }).max_by(|&(_, a), &(_, b)| {
            a.partial_cmp(&b).unwrap_or(Ordering::Equal)
        }) {
            Some(closest) => closest,
            None => panic!("Expected at least two clusters, found none")
        };

        {
            let cluster2 = clusters.remove(closest2);
            let cluster1 = clusters.get_mut(closest1).unwrap();
            cluster1.points.extend(cluster2.points);
        }

        clusters
    }

    fn merge_by_single_link(points: &[Point], mut clusters: Vec<Cluster>) -> Vec<Cluster> {
        let ((closest1, closest2), _) = match clusters.iter().enumerate().map(|(index_c1, cluster1)| {
            match clusters.iter().skip(index_c1 + 1).enumerate().map(|(index_c2, cluster2)| {
                let max_distance = match cluster1.points.iter().map(|point_c1| {
                    match cluster2.points.iter().map(|point_c2| {
                        SquaredEuclidean::distance(points[*point_c1].coordinates(), points[*point_c2].coordinates())
                    }).min_by(|a, b| {
                        a.partial_cmp(&b).unwrap_or(Ordering::Equal)
                    }) {
                        Some(closest) => closest,
                        None => f64::INFINITY
                    }
                }).min_by(|a, b| {
                    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
                }) {
                    Some(closest) => closest,
                    None => f64::INFINITY
                };

                ((index_c1, index_c2), max_distance)
            }).min_by(|&(_, a), &(_, b)| {
                a.partial_cmp(&b).unwrap_or(Ordering::Equal)
            }) {
                Some(closest) => closest,
                None => ((usize::max_value(), usize::max_value()), f64::INFINITY)
            }
        }).min_by(|&(_, a), &(_, b)| {
            a.partial_cmp(&b).unwrap_or(Ordering::Equal)
        }) {
            Some(closest) => closest,
            None => panic!("Expected at least two clusters, found none")
        };

        {
            let cluster2 = clusters.remove(closest2);
            let cluster1 = clusters.get_mut(closest1).unwrap();
            cluster1.points.extend(cluster2.points);
        }

        clusters
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
    fn bench_1000_points_agglomerative() {
        let mut rng = rand::thread_rng();
        let mut points: Vec<Point> = (0..1000).map(|_| {
            Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
        }).collect();

        let repeat_count = 10_u8;
        let mut total = 0_u64;
        for _ in 0..repeat_count {
            let start = time::precise_time_ns();
            Agglomerative::run(points.as_mut_slice(), 10, &Link::Centroid);
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