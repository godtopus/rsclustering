use rand;
use rand::distributions::{IndependentSample, Range};

use std::cmp::Ordering;
use std::usize;
use kmeanscluster::KMeansCluster;
use point::Point;
use distance::*;
use std::collections::HashMap;
use std::collections::BTreeMap;

pub struct KMeans {
    assignments: Vec<usize>,
    centroids: Vec<Point>,
    iterations: usize,
    converged: bool
}

impl KMeans {
    pub fn kmeans(points: &[Point], no_clusters: usize, max_iterations: usize) -> Self {
        let mut rng = rand::thread_rng();
        let between = Range::new(0, points.len());

        let mut centroids: BTreeMap<usize, Vec<f64>> = (0..no_clusters).map(|i| {
            let index = between.ind_sample(&mut rng);
            (i, points[index].coordinates().to_vec())
        }).collect();

        let mut previous_round: HashMap<usize, usize> = HashMap::with_capacity(points.len());

        let mut i = 1;

        while i < max_iterations {
            let mut has_converged = true;

            for (index_p, p) in points.iter().enumerate() {
                match centroids.iter().map(|(index_c, c)| {
                    (*index_c, SquaredEuclidean::distance(p.coordinates(), c))
                }).min_by(|&(_, a), &(_, b)| {
                    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
                }) {
                    Some((index_c, _)) => {
                        has_converged = previous_round.get(&index_p).unwrap_or(&usize::max_value()) == &index_c;
                        previous_round.insert(index_p, index_c);
                        ()
                    },
                    None => ()
                }
            }

            let mut new_centroids: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
            for (index_p, index_c) in previous_round.iter() {
                (*new_centroids.entry(*index_c).or_insert(vec![0.0; points[*index_p].coordinates().len()])).iter().zip(points[*index_p].coordinates().iter()).map(|(a, b)| a + b).collect::<Vec<f64>>();
            }

            centroids = new_centroids;

            if has_converged {
                break;
            }

            i += 1;
        }

        KMeans {
            assignments: previous_round.into_iter().collect::<BTreeMap<usize, usize>>().into_iter().map(|(_, p)| p).collect(),
            centroids: centroids.into_iter().map(|(_, c)| Point::new(c)).collect(),
            iterations: i,
            converged: i < max_iterations
        }
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
    fn bench_100000_points() {
        let mut rng = rand::thread_rng();
        let mut points: Vec<Point> = (0..100000).map(|_| Point::new(vec![rng.next_f64(), rng.next_f64()])).collect();

        let repeat_count = 10_u8;
        let mut total = 0_u64;
        for _ in 0..repeat_count {
            let start = time::precise_time_ns();
            KMeans::kmeans(points.as_mut_slice(), 10, 15);
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