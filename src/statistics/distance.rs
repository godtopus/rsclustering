use std::cmp::Ordering;
use nalgebra::*;
use statistics::statistics::Statistics;

pub struct SquaredEuclidean;
pub struct Euclidean;
pub struct Hamming;
pub struct Chebyshev;
pub struct Manhattan;
pub struct CosineSimilarity;
pub struct Mahalanobis;
pub struct Minkowski;

pub trait Distance {
    fn distance(_: &[f64], _: &[f64]) -> f64 {
        unimplemented!()
    }

    fn distance_with_parameter(_: &[f64], _: &[f64], _: f64) -> f64 { unimplemented!() }

    fn distance_with_covariance(_: &[f64], _: &[f64], _: &[&[f64]]) -> f64 { unimplemented!() }
}

impl Distance for SquaredEuclidean {
    #[inline]
    fn distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }
}

impl Distance for Euclidean {
    fn distance(a: &[f64], b: &[f64]) -> f64 {
        (a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>())
            .sqrt()
    }
}

impl Distance for Hamming {
    fn distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .filter(|&(x, y)| x != y)
            .count() as f64
    }
}

impl Distance for Chebyshev {
    fn distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .max_by(|x, y| x.partial_cmp(&y).unwrap_or(Ordering::Equal))
            .unwrap()
    }
}

impl Distance for Manhattan {
    #[inline]
    fn distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum()
    }
}

impl Distance for CosineSimilarity {
    #[inline]
    fn distance(a: &[f64], b: &[f64]) -> f64 {
        let (dot_product, magnitude_a, magnitude_b) =
            a.iter()
             .zip(b.iter())
             .fold((0.0, 0.0, 0.0), |(dp, m_a, m_b), (x, y)| (dp + x * y, m_a + x * x, m_b + y * y));

        dot_product / (magnitude_a.sqrt() * magnitude_b.sqrt())
    }
}

impl Distance for Mahalanobis {
    #[inline]
    fn distance_with_covariance(observation: &[f64], mean: &[f64], observations: &[&[f64]]) -> f64 {
        let column_vector = DMatrix::from_column_iter(observation.len(), 1, observation.iter().zip(mean.iter()).map(|(o, m)| o - m));
        let row_vector = column_vector.transpose();
        let inverse_covariance = Statistics::inverse_covariance_matrix(observations);

        (row_vector * inverse_covariance * column_vector).into_vector().into_iter().sum::<f64>().sqrt()
    }
}

impl Distance for Minkowski {
    fn distance_with_parameter(a: &[f64], b: &[f64], p: f64) -> f64 {
        (a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>())
            .powf(1.0 / p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_euclidean_is_correct_distance() {
        assert_eq!(false, true);
    }

    #[test]
    fn euclidean_is_correct_distance() {
        assert_eq!(false, true);
    }

    #[test]
    fn hamming_is_correct_distance() {
        let expected = 4.0;

        let input_a = vec![0.0, 1.0, 3.0, -8.7, 4.5, 1.0];
        let input_b = vec![-2.3, 1.0, -1.0, 3.0, 4.5, -2.3];

        let output = Hamming::distance(input_a.as_slice(), input_b.as_slice());

        assert_eq!(expected, output);
    }

    #[test]
    fn chebyshev_is_correct_distance() {
        assert_eq!(false, true);
    }

    #[test]
    fn manhattan_is_correct_distance() {
        assert_eq!(false, true);
    }
}