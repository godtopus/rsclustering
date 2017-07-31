use point::Point;
use distance::*;
use std::f64;
use std::ops::Deref;
use std::cmp::Ordering;

#[derive(Eq, PartialEq, Clone, Debug)]
pub struct KDTree {
    node: Option<Point>,
    left: Option<Box<KDTree>>,
    right: Option<Box<KDTree>>,
    axis: usize
}

impl KDTree {
    pub fn new(points: &mut [Point]) -> Self {
        match points.is_empty() {
            true => Self::empty(),
            false => *Self::new_with_depth(points, 0).unwrap()
        }
    }

    fn new_with_depth(points: &mut [Point], depth: usize) -> Option<Box<Self>> {
        match points.len() {
            0 => None,
            1 => {
                Self::new_node(points[0].clone(), depth % points[0].coordinates().len())
            }
            _ => {
                let axis = depth % points[0].coordinates().len();

                points.sort_by(|a, b| a.coordinates()[axis].partial_cmp(&b.coordinates()[axis]).unwrap());
                let median = points.len() / 2;

                Some(Box::new(KDTree {
                    node: Some(points[median].clone()),
                    left: Self::new_with_depth(&mut points[0..median], depth + 1),
                    right: Self::new_with_depth(&mut points[median + 1..], depth + 1),
                    axis: axis
                }))
            }
        }
    }

    fn empty() -> Self {
        KDTree {
            node: None,
            left: None,
            right: None,
            axis: 0
        }
    }

    fn new_node(point: Point, discriminator: usize) -> Option<Box<Self>> {
        Some(Box::new(KDTree {
            node: Some(point),
            left: None,
            right: None,
            axis: discriminator
        }))
    }

    pub fn insert(&mut self, point: &mut Point) {
        if self.node == None {
            *self = Self::new(&mut [point.clone()]);
            return
        }

        let mut nodes = vec![self];

        loop {
            let mut cur_node = nodes.pop().unwrap();

            match cur_node.node.as_ref() {
                Some(node) => {
                    if node.coordinates()[cur_node.axis] <= point.coordinates()[cur_node.axis] {
                        match cur_node.right {
                            Some(ref mut right) => nodes.push(right),
                            None => {
                                let discriminator = (cur_node.axis + 1) % point.coordinates().len();
                                cur_node.right = Self::new_node(point.clone(), discriminator);
                                return
                            }
                        }
                    } else {
                        match cur_node.left {
                            Some(ref mut left) => nodes.push(left),
                            None => {
                                let discriminator = (cur_node.axis + 1) % point.coordinates().len();
                                cur_node.left = Self::new_node(point.clone(), discriminator);
                                return
                            }
                        }
                    }
                },
                None => ()
            }
        }
    }

    pub fn remove(&mut self, point: &Point) -> bool {
        let mut nodes = vec![self];

        loop {
            let mut cur_node = nodes.pop().unwrap();
            let node = cur_node.node.clone().unwrap();

            if node.coordinates()[cur_node.axis] <= point.coordinates()[cur_node.axis] {
                match (node, point) {
                    (ref mut node, point) if node == point => {
                        Self::recursive_remove(cur_node);
                        return true;
                    },
                    _ => ()
                }

                match cur_node.right {
                    Some(ref mut right) => nodes.push(right),
                    None => ()
                }
            } else {
                match cur_node.left {
                    Some(ref mut left) => nodes.push(left),
                    None => ()
                }
            }

            if nodes.is_empty() {
                break;
            }
        }

        false
    }

    pub fn nearest_neighbor(&self, point: &Point) -> Option<Point> {
        match Self::nearest_neighbor_recursive(self, point, &mut None, f64::INFINITY) {
            Some((p, _)) => Some(p),
            _ => None
        }
    }

    fn nearest_neighbor_recursive(cur_node: &KDTree, point: &Point, best: &Option<Point>, best_distance: f64) -> Option<(Point, f64)> {
        let mut cur_best = best.clone();
        let mut cur_best_distance = best_distance;

        let distance = SquaredEuclidean::distance(point.coordinates(), cur_node.node.as_ref().unwrap().coordinates());
        if distance < cur_best_distance && cur_node.node.as_ref().unwrap() != point {
            cur_best = cur_node.node.clone();
            cur_best_distance = distance;
        }

        match (cur_node.left.as_ref(), cur_node.right.as_ref()) {
            (None, None) => {
                return Some((cur_best.unwrap(), cur_best_distance))
            },
            _ => {
                let node = cur_node.node.clone().unwrap();

                if node.coordinates()[cur_node.axis] <= point.coordinates()[cur_node.axis] {
                    if cur_node.right.is_some() && point.coordinates()[cur_node.axis] + cur_best_distance > node.coordinates()[cur_node.axis] {
                        match Self::nearest_neighbor_recursive(cur_node.right.as_ref().unwrap().deref(), point, &cur_best, cur_best_distance) {
                            Some((p, distance)) => {
                                if distance < cur_best_distance {
                                    cur_best = Some(p);
                                    cur_best_distance = distance;
                                }
                            },
                            _ => ()
                        }
                    }

                    if cur_node.left.is_some() && point.coordinates()[cur_node.axis] - cur_best_distance <= node.coordinates()[cur_node.axis] {
                        match Self::nearest_neighbor_recursive(cur_node.left.as_ref().unwrap().deref(), point, &cur_best, cur_best_distance) {
                            Some((p, distance)) => {
                                if distance < cur_best_distance {
                                    cur_best = Some(p);
                                    cur_best_distance = distance;
                                }
                            },
                            _ => ()
                        }
                    }
                } else {
                    if cur_node.left.is_some() && point.coordinates()[cur_node.axis] - cur_best_distance <= node.coordinates()[cur_node.axis] {
                        match Self::nearest_neighbor_recursive(cur_node.left.as_ref().unwrap().deref(), point, &cur_best, cur_best_distance) {
                            Some((p, distance)) => {
                                if distance < cur_best_distance {
                                    cur_best = Some(p);
                                    cur_best_distance = distance;
                                }
                            },
                            _ => ()
                        }
                    }

                    if cur_node.right.is_some() && point.coordinates()[cur_node.axis] + cur_best_distance > node.coordinates()[cur_node.axis] {
                        match Self::nearest_neighbor_recursive(cur_node.right.as_ref().unwrap().deref(), point, &cur_best, cur_best_distance) {
                            Some((p, distance)) => {
                                if distance < cur_best_distance {
                                    cur_best = Some(p);
                                    cur_best_distance = distance;
                                }
                            },
                            _ => ()
                        }
                    }
                }
            }
        }

        Some((cur_best.unwrap(), cur_best_distance))
    }

    pub fn inorder(&self) -> Vec<Point> {
        Self::recursive_inorder(&Some(self), vec![])
    }

    fn recursive_inorder(branch: &Option<&KDTree>, mut nodes: Vec<Point>) -> Vec<Point> {
        if *branch == None {
            return nodes
        }

        let cur_node = branch.unwrap().clone();

        match cur_node.left {
            Some(ref node) => nodes.append(&mut Self::recursive_inorder(&Some(&node), vec![])),
            None => ()
        }

        nodes.push(cur_node.node.unwrap());

        match cur_node.right {
            Some(ref node) => nodes.append(&mut Self::recursive_inorder(&Some(&node), vec![])),
            None => ()
        }

        nodes
    }

    fn recursive_remove(node_to_remove: &mut KDTree) {
        match (node_to_remove.left.as_ref(), node_to_remove.right.as_ref()) {
            (None, None) => {
                *node_to_remove = Self::empty();
                return
            },
            _ => ()
        }

        match node_to_remove.right {
            None => {
                node_to_remove.right = node_to_remove.left.clone();
                node_to_remove.left = None;
            },
            _ => ()
        }

        //println!("remove {:?}", node_to_remove);

        match Self::find_minimal_node(&node_to_remove.clone(), node_to_remove.axis) {
            Some((parent, Some(mut minimal_node))) => {
                match (parent.left, parent.right, node_to_remove.clone()) {
                    (Some(ref mut left), _, ref n) if (*left).deref() == n => Self::recursive_remove(left),
                    (_, Some(ref mut right), ref n) if (*right).deref() == n => Self::recursive_remove(right),
                    _ => ()
                }

                minimal_node.axis = node_to_remove.axis;
                minimal_node.left = node_to_remove.left.clone();

                *node_to_remove = minimal_node;

                //println!("minimal {:?}", node_to_remove);
            },
            _ => {
                *node_to_remove = Self::empty();
                return
            },
        };
    }

    fn find_minimal_node(node_head: &KDTree, discriminator: usize) -> Option<(Self, Option<Self>)> {
        let mut cur_node = (node_head, node_head.right.as_ref());
        let mut stack = vec![];
        let mut candidates = vec![];

        loop {
            cur_node = match cur_node {
                (_, Some(ref node)) => {
                    stack.push(cur_node);
                    (&node, node.left.as_ref())
                },
                (_, None) => {
                    match stack.len() {
                        0 => break,
                        _ => {
                            let temp = stack.pop().unwrap();
                            candidates.push(temp);

                            match cur_node {
                                (_, Some(ref node)) => (&node, node.right.as_ref()),
                                old => old
                            }
                        }
                    }
                }
            }
        }

        match candidates.into_iter().min_by(|&(_, x_c), &(_, y_c)| {
                x_c.unwrap().node.as_ref().unwrap().coordinates()[discriminator].partial_cmp(
                &y_c.unwrap().node.as_ref().unwrap().coordinates()[discriminator]).unwrap_or(Ordering::Equal) }) {
            Some((p, None)) => Some((p.clone(), None)),
            Some((p, Some(n))) => Some((p.clone(), Some(n.deref().clone()))),
            None => None
        }
    }
}

impl Ord for KDTree {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for KDTree {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.node.clone().unwrap_or(Point::new(vec![])).coordinates()[self.axis].partial_cmp(&other.node.clone().unwrap_or(Point::new(vec![])).coordinates()[other.axis])
    }
}

#[cfg(test)]
mod tests {
    //extern crate rand;

    use super::KDTree;
    use super::Point;

    /*fn random_point() -> ([f64; 2], i32) {
        rand::random::<([f64; 2], i32)>()
    }*/

    #[test]
    fn can_create_empty_kdtree() {
        let expected = KDTree {
            node: None,
            left: None,
            right: None,
            axis: 0
        };

        let kd_tree = KDTree::new(vec![].as_mut_slice());

        assert_eq!(expected, kd_tree);
    }

    #[test]
    fn can_create_multiple_node_kdtree() {
        let expected = KDTree {
            node: Some(Point::new(vec![2.0, 3.0])),
            left: Some(Box::new(KDTree {
                node: Some(Point::new(vec![1.0, 2.0])),
                left: Some(Box::new(KDTree {
                    node: Some(Point::new(vec![0.0, 1.0])),
                    left: None,
                    right: None,
                    axis: 0
                })),
                right: None,
                axis: 1
            })),
            right: Some(Box::new(KDTree {
                node: Some(Point::new(vec![4.0, 5.0])),
                left: Some(Box::new(KDTree {
                    node: Some(Point::new(vec![3.0, 4.0])),
                    left: None,
                    right: None,
                    axis: 0
                })),
                right: None,
                axis: 1
            })),
            axis: 0
        };

        let kd_tree = KDTree::new(vec![Point::new(vec![0.0, 1.0]), Point::new(vec![1.0, 2.0]), Point::new(vec![2.0, 3.0]), Point::new(vec![3.0, 4.0]), Point::new(vec![4.0, 5.0])].as_mut_slice());

        assert_eq!(expected, kd_tree);
    }

    #[test]
    fn can_insert_into_kdtree() {
        let expected = KDTree {
            node: Some(Point::new(vec![0.0, 1.0])),
            left: Some(Box::new(KDTree {
                node: Some(Point::new(vec![-1.0, 0.0])),
                left: None,
                right: None,
                axis: 1
            })),
            right: Some(Box::new(KDTree {
                node: Some(Point::new(vec![1.0, 2.0])),
                left: None,
                right: Some(Box::new(KDTree {
                    node: Some(Point::new(vec![2.0, 3.0])),
                    left: None,
                    right: None,
                    axis: 0
                })),
                axis: 1
            })),
            axis: 0
        };

        let mut kd_tree = KDTree::new(vec![].as_mut_slice());
        kd_tree.insert(&mut Point::new(vec![0.0, 1.0]));
        kd_tree.insert(&mut Point::new(vec![1.0, 2.0]));
        kd_tree.insert(&mut Point::new(vec![-1.0, 0.0]));
        kd_tree.insert(&mut Point::new(vec![2.0, 3.0]));

        assert_eq!(expected, kd_tree);
    }

    #[test]
    fn can_remove_node_from_single_node_kdtree() {
        let expected = KDTree {
            node: None,
            left: None,
            right: None,
            axis: 0
        };

        let mut kd_tree = KDTree::new(vec![Point::new(vec![0.0, 1.0])].as_mut_slice());
        kd_tree.remove(&Point::new(vec![0.0, 1.0]));

        assert_eq!(expected, kd_tree);
    }

    #[test]
    fn can_remove_node_from_multiple_node_kdtree() {
        let expected = KDTree {
            node: Some(Point::new(vec![2.0, 3.0])),
            left: Some(Box::new(KDTree {
                node: Some(Point::new(vec![1.0, 2.0])),
                left: Some(Box::new(KDTree {
                    node: Some(Point::new(vec![0.0, 1.0])),
                    left: None,
                    right: None,
                    axis: 0
                })),
                right: None,
                axis: 1
            })),
            right: Some(Box::new(KDTree {
                node: Some(Point::new(vec![3.0, 4.0])),
                left: None,
                right: None,
                axis: 1
            })),
            axis: 0
        };

        let mut kd_tree = KDTree::new(vec![Point::new(vec![0.0, 1.0]), Point::new(vec![1.0, 2.0]), Point::new(vec![2.0, 3.0]), Point::new(vec![3.0, 4.0]), Point::new(vec![4.0, 5.0])].as_mut_slice());
        kd_tree.remove(&Point::new(vec![4.0, 5.0]));

        assert_eq!(expected, kd_tree);
    }

    #[test]
    fn can_find_nearest_neighbor_kdtree() {
        let expected = Point::new(vec![2.0, 3.0]);

        let kd_tree = KDTree::new(vec![Point::new(vec![0.0, 1.0]), Point::new(vec![2.0, 3.0]), Point::new(vec![3.0, 4.0])].as_mut_slice());
        let nearest_neighbor = kd_tree.nearest_neighbor(&Point::new(vec![3.0, 4.0]));

        assert_eq!(expected, nearest_neighbor.unwrap());
    }

    #[test]
    fn can_traverse_tree_inorder() {
        let expected = vec![Point::new(vec![0.0, 1.0]), Point::new(vec![1.0, 2.0]), Point::new(vec![2.0, 3.0]), Point::new(vec![3.0, 4.0]), Point::new(vec![4.0, 5.0])];

        let kd_tree = KDTree::new(vec![Point::new(vec![0.0, 1.0]), Point::new(vec![1.0, 2.0]), Point::new(vec![2.0, 3.0]), Point::new(vec![3.0, 4.0]), Point::new(vec![4.0, 5.0])].as_mut_slice());

        assert_eq!(expected, kd_tree.inorder());
    }
}