pub struct SquaredEuclidean;

pub trait Distance {
    fn distance(_: &[f64], _: &[f64]) -> f64 {
        0.0
    }
}

impl Distance for SquaredEuclidean {
    fn distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }
}