use std::cmp::Ordering;
use std::f64;
use std::ops::Deref;
use point::Point;
use distance::*;
use std::fmt::Debug;

#[derive(Clone)]
pub struct Cluster<T: Clone + Copy + Debug> {
    distance: f64,
    mean: Vec<f64>,
    points: Vec<Point<T>>,
    representative: Vec<Point<T>>,
    closest: Option<Box<Cluster<T>>>
}

impl <T: Clone + Copy + Debug> Cluster<T> {
    pub fn new(point: Point<T>) -> Cluster<T> {
        Cluster {
            distance: f64::INFINITY,
            mean: point.coordinates().to_vec(),
            points: vec![point.clone()],
            representative: vec![point.clone()],
            closest: None
        }
    }

    pub fn merge(&self, number_represent_points: usize, compression: f64) -> Option<Cluster<T>> {
        let mut merged_points = self.points.clone();
        let b = match self.closest.clone() {
            None => return None,
            Some(cluster) => {
                merged_points.extend(cluster.clone().points);
                cluster.deref().clone()
            }
        };

        let a_len = self.points.len() as f64;
        let b_len = b.points.len() as f64;

        let merged_mean = match (merged_points.first(), merged_points.last()) {
            (Some(first), Some(last)) if *first == *last => first.coordinates().to_vec(),
            _ => (0..self.mean.len()).map(|i| {
                (a_len * self.mean.get(i).unwrap_or(&0.0) + b_len * b.mean.get(i).unwrap_or(&0.0)) / (a_len + b_len)
            }).collect::<Vec<f64>>()
        };

        let mut temp: Vec<Point<T>> = vec![];
        for index in 0..number_represent_points {
            let mut maximal_distance = 0.0;
            let mut maximal_point = None;

            for point in merged_points.clone() {
                let minimal_distance = match index {
                    0 => SquaredEuclidean::distance(point.coordinates(), &merged_mean),
                    _ => temp.iter().map(|p| SquaredEuclidean::distance(point.coordinates(), p.coordinates())).fold(f64::INFINITY, f64::min)
                };

                if minimal_distance > maximal_distance {
                    maximal_distance = minimal_distance;
                    maximal_point = Some(point);
                }
            }

            match maximal_point {
                Some(point) => {
                    if !temp.contains(&point) {
                        temp.push(point);
                    }
                },
                None => ()
            }
        }

        let merged_rep = temp.iter().map(|p| {
            let coords = p.coordinates();
            Point::new((0..coords.len()).map(|i| coords.get(i).unwrap() + compression * (merged_mean.get(i).unwrap() - coords.get(i).unwrap())).collect())
        }).collect();

        Some(Cluster {
            distance: 0.0,
            mean: merged_mean.to_vec(),
            points: merged_points,
            representative: merged_rep,
            closest: None
        })
    }

    pub fn update_closest(&mut self) {

    }
}

impl <T: Clone + Copy + Debug> Ord for Cluster<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl <T: Clone + Copy + Debug> PartialOrd for Cluster<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl <T: Clone + Copy + Debug> Eq for Cluster<T> {}

impl <T: Clone + Copy + Debug> PartialEq for Cluster<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}