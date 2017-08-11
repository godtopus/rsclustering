use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use std::cmp::Ordering;
use std::usize;
use point::Point;
use distance::*;
use statistics::Statistics;
use kmeans::*;

pub struct MiniBatchKMeans {
    assignments: Vec<usize>,
    centroids: Vec<Point>,
    iterations: usize,
    converged: bool
}

impl MiniBatchKMeans {
    pub fn run(points: &[Point], no_clusters: usize, max_iterations: usize, tolerance: f64, batch_size: usize, init_method: KMeansInitialization, precomputed: Option<&[Vec<f64>]>) -> Self {
        let mut centroids = KMeans::initial_centroids(points, no_clusters, init_method, precomputed);
        let mut cluster_size = vec![0; no_clusters];

        let mut rng = rand::thread_rng();
        let between = Range::new(0, points.len());

        let mut i = 0;
        let stop_condition = tolerance * tolerance;

        while i < max_iterations {
            let previous_centroids = centroids.clone();

            for _ in 0..batch_size {
                let p = points[between.ind_sample(&mut rng)].coordinates();

                let (index_c, _) =  Self::closest_centroid(p, centroids.as_slice());
                cluster_size[index_c] += 1;


                // Gradient descent
                let eta = 1.0 / (cluster_size[index_c]) as f64;
                let eta_compliment = 1.0 - eta;
                centroids[index_c] = centroids[index_c].iter().zip(p.iter()).map(|(c, p)| {
                    eta_compliment * c + eta * p
                }).collect();
            };

            let change = match previous_centroids.iter().zip(centroids.iter()).map(|(centroid, updated_centroid)| {
                SquaredEuclidean::distance(&centroid, &updated_centroid)
            }).max_by(|a, b| {
                a.partial_cmp(&b).unwrap_or(Ordering::Equal)
            }) {
                Some(max_change) => max_change,
                None => panic!()
            };

            if change < stop_condition {
                break;
            }

            i += 1;
        }

        MiniBatchKMeans {
            assignments: vec![],
            centroids: centroids.into_iter().map(|c| Point::new(c)).collect(),
            iterations: i,
            converged: i < max_iterations
        }
    }

    #[inline]
    fn closest_centroid(point: &[f64], centroids: &[Vec<f64>]) -> (usize, f64) {
        match centroids.iter().enumerate().map(|(index_c, c)| {
            (index_c, SquaredEuclidean::distance(point, c))
        }).min_by(|&(_, a), &(_, b)| {
            a.partial_cmp(&b).unwrap_or(Ordering::Equal)
        }) {
            Some(closest) => closest,
            None => panic!()
        }
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
    fn bench_100000_points_mini_batch_kmeans() {
        let mut rng = rand::thread_rng();
        let mut points: Vec<Point> = (0..100000).map(|_| {
            Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
        }).collect();

        let repeat_count = 10_u8;
        let mut total = 0_u64;
        for _ in 0..repeat_count {
            let start = time::precise_time_ns();
            MiniBatchKMeans::run(points.as_mut_slice(), 10, 15, 0.00001, 10000, KMeansInitialization::Random, None);
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