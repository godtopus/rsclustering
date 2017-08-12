#![feature(test)]

extern crate rust_clustering;
extern crate test;
extern crate rand;

use rust_clustering::clustering::agglomerative::*;
use rust_clustering::clustering::fuzzy_cmeans::*;
use rust_clustering::clustering::kmeans::*;
use rust_clustering::clustering::kmedians::*;
use rust_clustering::clustering::kmedoids::*;
use rust_clustering::clustering::mini_batch_kmeans::*;
use rust_clustering::point::*;
use test::Bencher;
use rand::*;

#[bench]
fn bench_1000_points_agglomerative(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let mut points: Vec<Point> = (0..1000).map(|_| {
        Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
    }).collect();

    b.iter(|| {
        Agglomerative::run(points.as_mut_slice(), 10, &Link::Centroid)
    });
}

#[bench]
fn bench_100000_points_fuzzy_cmeans(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let mut points: Vec<Point> = (0..100000).map(|_| {
        Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
    }).collect();

    b.iter(|| {
        FuzzyCMeans::run(points.as_mut_slice(), 10, 15, 2.0, 0.00001, FuzzyCMeansInitialization::Random, None);
    });
}

#[bench]
fn bench_100000_points_kmeans(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let mut points: Vec<Point> = (0..100000).map(|_| {
        Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
    }).collect();

    b.iter(|| {
        KMeans::new().run(points.as_mut_slice(), 10)
    });
}

#[bench]
fn bench_100000_points_kmedians(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let mut points: Vec<Point> = (0..100000).map(|_| {
        Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
    }).collect();

    b.iter(|| {
        KMedians::new().run(points.as_mut_slice(), 10)
    });
}

#[bench]
fn bench_100000_points_kmedoids(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let mut points: Vec<Point> = (0..100000).map(|_| {
        Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
    }).collect();

    b.iter(|| {
        KMedoids::new().run(points.as_mut_slice(), 10)
    });
}

#[bench]
fn bench_100000_points_mini_batch_kmeans(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let mut points: Vec<Point> = (0..100000).map(|_| {
        Point::new((0..2).into_iter().map(|_| rng.next_f64()).collect())
    }).collect();

    b.iter(|| {
        MiniBatchKMeans::new().run(points.as_mut_slice(), 10);
    });
}