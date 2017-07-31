use test::Bencher;

#[bench]
fn bench_100000_points(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let points = Vec::from_fn(100000, |_| Point::new(vec![rng.next_f64(), rng.next_f64()]));

    b.iter(|| {
        kmeans(points, 50, usize::max_value());
    });
}