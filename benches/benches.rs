#![feature(test)]

extern crate rust_clustering;
extern crate test;
extern crate rand;

use rust_clustering::clustering::kmeans::*;
use rust_clustering::point::*;
use test::Bencher;
use rand::*;

#[bench]
fn bench_100000_points(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let mut points: Vec<Point> = (0..100000).map(|_| {
        Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
    }).collect();

    b.iter(|| {
        KMeans::new().run(points.as_mut_slice(), 10)
    });
}