use std::cmp::Ordering;
use std::usize;
use std::f64;
use point::Point;
use clustering::agglomerative::Link::*;
use statistics::distance::{Distance, SquaredEuclidean};
use statistics::statistics::Statistics;
use rayon::prelude::*;

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
        let mut clusters: Vec<Cluster> = match *link_criterion {
            Single | Complete | Average =>
                (0..points.len()).map(|p| {
                    Cluster {
                        points: vec![p],
                        centroid: vec![]
                    }
                }).collect(),
            Centroid =>
                points.iter().enumerate().map(|(index, p)| {
                    Cluster {
                        points: vec![index],
                        centroid: p.coordinates().to_vec()
                    }
                }).collect(),
        };

        let mut i = 0;

        while clusters.len() > no_clusters {
            clusters = Self::merge_clusters(points, clusters, link_criterion);
            i += 1;
        }

        Agglomerative {
            assignments: vec![],
            centroids: vec![],
            iterations: i,
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
        let ((closest1, closest2), _) = match clusters.par_iter().enumerate().map(|(index_c1, cluster1)| {
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
        let ((closest1, closest2), _) = match clusters.par_iter().enumerate().map(|(index_c1, cluster1)| {
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
        let ((closest1, closest2), _) = match clusters.par_iter().enumerate().map(|(index_c1, cluster1)| {
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
        let ((closest1, closest2), _) = match clusters.par_iter().enumerate().map(|(index_c1, cluster1)| {
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