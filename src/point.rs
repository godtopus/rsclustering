use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct Point<T: Clone + Copy + Debug> {
    coordinates: Vec<f64>,
    data: Option<T>
}

impl <T: Clone + Copy + Debug> Point<T> {
    pub fn new(coordinates: Vec<f64>) -> Self {
        Point {
            coordinates: coordinates,
            data: None
        }
    }

    pub fn new_with_data(coordinates: Vec<f64>, data: T) -> Self {
        Point {
            coordinates: coordinates,
            data: Some(data)
        }
    }

    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }
}

impl <T: Clone + Copy + Debug> Eq for Point<T> {}

impl <T: Clone + Copy + Debug> PartialEq for Point<T> {
    fn eq(&self, other: &Self) -> bool {
        self.coordinates.len() == other.coordinates.len() && self.coordinates.iter().zip(other.coordinates.iter()).all(|(x, y)| x == y)
    }
}