use super::Dataset;
use point::Point;

pub fn load() -> Dataset<Vec<Point>, Vec<f64>> {
    let data = vec![Point::new(vec![8.3, 70.]),
                    Point::new(vec![8.6, 65.]),
                    Point::new(vec![8.8, 63.]),
                    Point::new(vec![10.5, 72.]),
                    Point::new(vec![10.7, 81.]),
                    Point::new(vec![10.8, 83.]),
                    Point::new(vec![11.0, 66.]),
                    Point::new(vec![11.0, 75.]),
                    Point::new(vec![11.1, 80.]),
                    Point::new(vec![11.2, 75.]),
                    Point::new(vec![11.3, 79.]),
                    Point::new(vec![11.4, 76.]),
                    Point::new(vec![11.4, 76.]),
                    Point::new(vec![11.7, 69.]),
                    Point::new(vec![12.0, 75.]),
                    Point::new(vec![12.9, 74.]),
                    Point::new(vec![12.9, 85.]),
                    Point::new(vec![13.3, 86.]),
                    Point::new(vec![13.7, 71.]),
                    Point::new(vec![13.8, 64.]),
                    Point::new(vec![14.0, 78.]),
                    Point::new(vec![14.2, 80.]),
                    Point::new(vec![14.5, 74.]),
                    Point::new(vec![16.0, 72.]),
                    Point::new(vec![16.3, 77.]),
                    Point::new(vec![17.3, 81.]),
                    Point::new(vec![17.5, 82.]),
                    Point::new(vec![17.9, 80.]),
                    Point::new(vec![18.0, 80.]),
                    Point::new(vec![18.0, 80.]),
                    Point::new(vec![20.6, 87.])];

    let target = vec![10.3, 10.3, 10.2, 16.4, 18.8, 19.7, 15.6, 18.2, 22.6, 19.9,
                      24.2, 21.0, 21.4, 21.3, 19.1, 22.2, 33.8, 27.4, 25.7, 24.9,
                      34.5, 31.7, 36.3, 38.3, 42.6, 55.4, 55.7, 58.3, 51.5, 51.0,
                      77.0];
    Dataset {
        data,
        target
    }
}