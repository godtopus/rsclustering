use distance::*;
use point::Point;
use std::f64::consts::PI;

pub struct Statistics;

impl Statistics {
    #[inline]
    pub fn mean(centroids: &[&[f64]]) -> Vec<f64> {
        #[inline]
        match centroids.len() {
            0 => vec![],
            _ => {
                let dimension = centroids[0].len() as f64;

                centroids.iter().fold(vec![0.0; centroids[0].len()], |mut acc, next| {
                    for i in 0..next.len() {
                        acc[i] += next[i];
                    }

                    acc
                }).into_iter().map(|x| x / dimension).collect()
            }
        }
    }

    pub fn variance(centroid: &[f64], points: &[Point]) -> f64 {
        points.iter().map(|p| SquaredEuclidean::distance(centroid, p.coordinates())).sum()
    }

    /**
     * Calculates the BIC for single cluster.
     * @param n the total number of samples.
     * @param d the dimensionality of data.
     * @param distortion the distortion of clusters (i.e. variance).
     * @return the BIC score.
     */
    fn bic_single(n: f64, d: f64, distortion: f64) -> f64 {
        let variance = distortion / (n - 1.0);

        let p1 = -n * (2.0 * PI).ln();
        let p2 = -n * d * variance.ln();
        let p3 = -(n - 1.0);

        ((p1 + p2 + p3) / 2.0) - 0.5 * (d + 1.0) * n.ln()
    }

    /**
     * Calculates the BIC for the given set of centers.
     * @param k the number of clusters.
     * @param n the total number of samples.
     * @param d the dimensionality of data.
     * @param distortion the distortion of clusters.
     * @param cluster_size the number of samples in each cluster.
     * @return the BIC score.
     */
    fn bic(k: f64, n: f64, d: f64, distortion: f64, cluster_size: &[f64]) -> f64 {
        let variance = distortion / (n - k);

        (0..k as usize).into_iter().map(|i| Self::log_likelihood(k, n, cluster_size[i], d, variance)).sum::<f64>() - 0.5 * (k + k * d) * n.ln()
    }

    /**
     * Estimate the log-likelihood of the data for the given model.
     *
     * @param k the number of clusters.
     * @param n the total number of samples.
     * @param ni the number of samples belong to this cluster.
     * @param d the dimensionality of data.
     * @param variance the estimated variance of clusters.
     * @return the likelihood estimate
     */
    fn log_likelihood(k: f64, n: f64, ni: f64, d: f64, variance:f64) -> f64 {
        let p1 = -ni * (2.0 * PI).ln();
        let p2 = -ni * d * variance.ln();
        let p3 = -(ni - k);
        let p4 = ni * ni.ln();
        let p5 = -ni * n.ln();
        (p1 + p2 + p3) / 2.0 + p4 + p5
    }
}