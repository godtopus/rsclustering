use point::Point;
use distance::*;

#[derive(Clone, Debug)]
pub struct KMeansCluster {
    points: Vec<Point>,
    mean: Vec<f64>
}

impl KMeansCluster {
    pub fn new(point: Point) -> Self {
        KMeansCluster {
            mean: point.coordinates().to_vec(),
            points: vec![point]
        }
    }

    pub fn from(points: Vec<Point>) -> Self {
        let mut cluster = KMeansCluster {
            mean: vec![0.0; points[0].coordinates().len()],
            points: points
        };

        cluster.update_mean();
        cluster
    }

    pub fn merge(&mut self, point: Point) {
        self.points.push(point);
    }

    pub fn distance(&self, point: &Point) -> f64 {
        SquaredEuclidean::distance(&self.mean, point.coordinates())
    }

    pub fn update_mean(&mut self) {
        let dimension = self.mean.len() as f64;
        self.mean = self.points.iter().fold(vec![0.0; dimension as usize], |mean, point| {
            point.coordinates().iter().zip(mean.iter()).map(|(p, m)| p + m).collect()
        }).iter().map(|m| m / dimension).collect()
    }

    pub fn points(&self) -> Vec<Point> {
        self.points.clone()
    }

    pub fn clear(&mut self) {
        self.points = vec![];
    }
}

impl Eq for KMeansCluster {}

impl PartialEq for KMeansCluster {
    fn eq(&self, other: &Self) -> bool {
        self.points.iter().zip(other.points.iter()).all(|(a, b)| a == b)
    }
}