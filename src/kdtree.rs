use point::Point;
use std::ops::Deref;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::DerefMut;

#[derive(Eq, PartialEq, Clone, Debug)]
pub struct KDTree<T: Clone + Copy + Debug + PartialEq + Eq> {
    node: Option<Point<T>>,
    left: Box<Option<KDTree<T>>>,
    right: Box<Option<KDTree<T>>>,
    axis: usize
}

impl <T: Clone + Copy + Debug + PartialEq + Eq> KDTree<T> {
    pub fn new(points: &mut [Point<T>]) -> Self {
        if points.is_empty() {
            return KDTree {
                node: None,
                left: Box::new(None),
                right: Box::new(None),
                axis: 0
            }
        }

        Self::new_with_depth(points, 0).unwrap()
    }

    fn new_with_depth(points: &mut [Point<T>], depth: usize) -> Option<Self> {
        match points.len() {
            0 => None,
            1 => {
                Some(KDTree {
                    node: Some(points[0].clone()),
                    left: Box::new(None),
                    right: Box::new(None),
                    axis: depth % points[0].coordinates().len()
                })
            }
            _ => {
                let axis = depth % points[0].coordinates().len();

                points.sort_by(|a, b| a.coordinates()[axis].partial_cmp(&b.coordinates()[axis]).unwrap());
                let median = points.len() / 2;

                Some(KDTree {
                    node: Some(points[median].clone()),
                    left: Box::new(Self::new_with_depth(&mut points[0..median], depth + 1)),
                    right: Box::new(Self::new_with_depth(&mut points[median + 1..], depth + 1)),
                    axis: axis
                })
            }
        }
    }

    fn new_node(point: Point<T>, discriminator: usize) -> Box<Option<Self>> {
        Box::new(Some(KDTree {
            node: Some(point),
            left: Box::new(None),
            right: Box::new(None),
            axis: discriminator
        }))
    }

    pub fn insert(&mut self, point: &mut Point<T>) {
        if self.node == None {
            *self = Self::new(&mut [point.clone()]);
            return
        }

        let mut cur_node = self;

        loop {
            if cur_node.node.clone().unwrap().coordinates()[cur_node.axis] <= point.coordinates()[cur_node.axis] {
                match cur_node.right.take() {
                    Some(ref mut right) => cur_node = right,
                    None => {
                        let discriminator = (cur_node.axis + 1) % point.coordinates().len();
                        cur_node.right = Self::new_node(point.clone(), discriminator);
                        return
                    }
                }
            } else {
                match cur_node.left.take() {
                    Some(ref mut left) => cur_node = left,
                    None => {
                        let discriminator = (cur_node.axis + 1) % point.coordinates().len();
                        cur_node.left = Self::new_node(point.clone(), discriminator);
                        return
                    }
                }
            }
        }
    }

    pub fn remove(&mut self, point: &mut Point<T>) -> bool {
        let (parent, mut node_to_remove) = match self.find_node(point) {
            (p, Some(node)) => (p, node),
            _ => return false
        };

        let minimal_node = self.recursive_remove(&mut node_to_remove);
        match parent {
            Some(mut node) => {
                match (node.left.take(), node.right.take(), node_to_remove) {
                    (Some(ref left), _, ref n) if left == n => node.left = Box::new(minimal_node),
                    (_, Some(ref right), ref n) if right == n => node.right = Box::new(minimal_node),
                    _ => ()
                }
            },
            None => *self = minimal_node.unwrap_or(Self::new(&mut vec![]))
        }

        true
    }

    /*pub fn nearest_neighbor(&mut self, point: &mut Point<T>) -> Option<Point<T>> {
        None
    }*/

    pub fn inorder(&self) -> Vec<Point<T>> {
        Self::recursive_inorder(&Some(self), vec![])
    }

    fn recursive_inorder(branch: &Option<&KDTree<T>>, mut nodes: Vec<Point<T>>) -> Vec<Point<T>> {
        if *branch == None {
            return nodes
        }

        let cur_node = branch.unwrap().clone();

        match cur_node.left.deref() {
            &Some(ref node) => nodes.append(&mut Self::recursive_inorder(&Some(&node), vec![])),
            &None => ()
        }

        nodes.push(cur_node.node.unwrap());

        match cur_node.right.deref() {
            &Some(ref node) => nodes.append(&mut Self::recursive_inorder(&Some(&node), vec![])),
            &None => ()
        }

        nodes
    }

    fn find_node(&self, point: &Point<T>) -> (Option<Self>, Option<Self>) {
        let mut found = (None, None);

        if self.node == None {
            return found
        }

        let mut cur_node: (Option<&Self>, &Self) = (None, self);

        loop {
            let node = cur_node.1.node.clone().unwrap();

            if node.coordinates()[cur_node.1.axis] <= point.coordinates()[cur_node.1.axis] {
                match (node, point) {
                    (ref node, point) if node == point => {
                        found = match cur_node {
                            (Some(p), n) => (Some(p.clone()), Some(n.clone())),
                            (None, n) => (None, Some(n.clone()))
                        };
                        break;
                    },
                    _ => ()
                }

                match *cur_node.1.right.deref() {
                    Some(ref right) => cur_node = (Some(cur_node.1), right),
                    None => ()
                }
            } else {
                match *cur_node.1.left.deref() {
                    Some(ref left) => cur_node = (Some(cur_node.1), left),
                    None => ()
                }
            }
        }

        found
    }

    fn recursive_remove(&self, node_to_remove: &mut KDTree<T>) -> Option<Self> {
        match (node_to_remove.left.deref(), node_to_remove.right.deref()) {
            (&None, &None) => return None,
            _ => ()
        }

        let discriminator = node_to_remove.axis;
        match *node_to_remove.right.deref() {
            None => {
                node_to_remove.right = node_to_remove.left.clone();
                node_to_remove.left = Box::new(None);
            },
            _ => ()
        }

        let (mut parent, mut minimal_node) = match self.find_minimal_node(&node_to_remove, discriminator) {
            Some((p, Some(node))) => (p, node),
            _ => return None
        };

        match (parent.left.take(), parent.right.take(), node_to_remove.clone()) {
            (Some(ref left), _, ref n) if left == n => parent.left = Box::new(self.recursive_remove(&mut minimal_node)),
            (_, Some(ref right), ref n) if right == n => parent.right = Box::new(self.recursive_remove(&mut minimal_node)),
            _ => ()
        }

        minimal_node.axis = node_to_remove.axis;
        minimal_node.right = node_to_remove.right.clone();
        minimal_node.left = node_to_remove.left.clone();

        Some(minimal_node)
    }

    fn find_minimal_node(&self, node_head: &KDTree<T>, discriminator: usize) -> Option<(Self, Option<Self>)> {
        let mut cur_node = (node_head, node_head.right.deref());
        let mut stack = vec![];
        let mut candidates = vec![];

        loop {
            match cur_node {
                (_, &Some(ref node)) => {
                    stack.push(cur_node);
                    cur_node = (&node, node.left.deref());
                },
                (_, &None) => {
                    match stack.len() {
                        0 => break,
                        _ => {
                            cur_node = stack.pop().unwrap();
                            candidates.push(cur_node);

                            cur_node = match cur_node {
                                (_, &Some(ref node)) => (&node, node.right.deref()),
                                old => old
                            };
                        }
                    }
                }
            }
        }

        match candidates.into_iter().min_by(|&(_, x_c), &(_, y_c)| {
                x_c.clone().unwrap().node.unwrap().coordinates()[discriminator].partial_cmp(
                &y_c.clone().unwrap().node.unwrap().coordinates()[discriminator]).unwrap_or(Ordering::Equal) }) {
            Some((p, &None)) => Some((p.clone(), None)),
            Some((p, &Some(ref n))) => Some((p.clone(), Some(n.clone()))),
            None => None
        }
    }
}

impl <T: Clone + Copy + Debug + Eq + PartialEq> Ord for KDTree<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl <T: Clone + Copy + Debug + Eq + PartialEq> PartialOrd for KDTree<T> {
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
            left: Box::new(None),
            right: Box::new(None),
            axis: 0
        };

        let kd_tree = KDTree::<u32>::new(vec![].as_mut_slice());

        assert_eq!(expected, kd_tree);
    }

    #[test]
    fn can_create_multiple_node_kdtree() {
        let expected = KDTree {
            node: Some(Point::new(vec![2.0, 3.0])),
            left: Box::new(Some(KDTree {
                node: Some(Point::new(vec![1.0, 2.0])),
                left: Box::new(Some(KDTree {
                    node: Some(Point::new(vec![0.0, 1.0])),
                    left: Box::new(None),
                    right: Box::new(None),
                    axis: 0
                })),
                right: Box::new(None),
                axis: 1
            })),
            right: Box::new(Some(KDTree {
                node: Some(Point::new(vec![4.0, 5.0])),
                left: Box::new(Some(KDTree {
                    node: Some(Point::new(vec![3.0, 4.0])),
                    left: Box::new(None),
                    right: Box::new(None),
                    axis: 0
                })),
                right: Box::new(None),
                axis: 1
            })),
            axis: 0
        };

        let kd_tree = KDTree::<u32>::new(vec![Point::new(vec![0.0, 1.0]), Point::new(vec![1.0, 2.0]), Point::new(vec![2.0, 3.0]), Point::new(vec![3.0, 4.0]), Point::new(vec![4.0, 5.0])].as_mut_slice());

        assert_eq!(expected, kd_tree);
    }

    #[test]
    fn can_insert_into_kdtree() {
        let expected = KDTree {
            node: Some(Point::new(vec![0.0, 1.0])),
            left: Box::new(Some(KDTree {
                node: Some(Point::new(vec![-1.0, 0.0])),
                left: Box::new(None),
                right: Box::new(None),
                axis: 1
            })),
            right: Box::new(Some(KDTree {
                node: Some(Point::new(vec![1.0, 2.0])),
                left: Box::new(None),
                right: Box::new(Some(KDTree {
                    node: Some(Point::new(vec![2.0, 3.0])),
                    left: Box::new(None),
                    right: Box::new(None),
                    axis: 1
                })),
                axis: 1
            })),
            axis: 0
        };

        let mut kd_tree = KDTree::<u32>::new(vec![].as_mut_slice());
        kd_tree.insert(&mut Point::new(vec![0.0, 1.0]));
        kd_tree.insert(&mut Point::new(vec![1.0, 2.0]));
        kd_tree.insert(&mut Point::new(vec![-1.0, 0.0]));
        kd_tree.insert(&mut Point::new(vec![2.0, 3.0]));

        assert_eq!(expected, kd_tree);
    }

    #[test]
    fn can_traverse_tree_inorder() {
        let expected = vec![Point::new(vec![0.0, 1.0]), Point::new(vec![1.0, 2.0]), Point::new(vec![2.0, 3.0]), Point::new(vec![3.0, 4.0]), Point::new(vec![4.0, 5.0])];

        let kd_tree = KDTree::<u32>::new(vec![Point::new(vec![0.0, 1.0]), Point::new(vec![1.0, 2.0]), Point::new(vec![2.0, 3.0]), Point::new(vec![3.0, 4.0]), Point::new(vec![4.0, 5.0])].as_mut_slice());

        assert_eq!(expected, kd_tree.inorder());
    }
}