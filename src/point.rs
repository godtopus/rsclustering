use std::hash::Hash;
use std::mem;
use std::hash::Hasher;

#[derive(Clone, Debug)]
pub struct Point {
    coordinates: Vec<f64>,
}

impl Point {
    pub fn new(coordinates: Vec<f64>) -> Self {
        Point {
            coordinates: coordinates
        }
    }

    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }
}

impl Eq for Point {}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.coordinates.len() == other.coordinates.len() && self.coordinates.iter().zip(other.coordinates.iter()).all(|(x, y)| x == y)
    }
}

impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Perform a bit-wise transform, relying on the fact that we
        // are never Infinity or NaN

        for coord in self.coordinates.iter() {
            state.write_u64(unsafe { mem::transmute(coord) });
        }

        state.finish();
    }
}