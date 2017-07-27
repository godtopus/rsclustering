use point::Point;
use std::ops::Deref;
use std::cmp::Ordering;

#[derive(Eq, PartialEq)]
pub struct KDTree<T: Clone + PartialEq + Eq> {
    node: Option<Point<T>>,
    left: Box<Option<KDTree<T>>>,
    right: Box<Option<KDTree<T>>>,
    axis: usize
}

impl <T: Clone + PartialEq + Eq> KDTree<T> {
    pub fn new(points: &mut [Point<T>]) -> Self {
        if points.is_empty() {
            return KDTree {
                node: None,
                left: Box::new(None),
                right: Box::new(None),
                axis: 0
            }
        }

        Self::new_with_depth(&mut points, 0).unwrap()
    }

    fn new_with_depth(points: &mut [Point<T>], depth: usize) -> Option<Self> {
        if points.is_empty() {
            return None
        }

        let k = points.get(0).unwrap_or(&Point::new(vec![])).coordinates().len();
        let axis = depth % k;

        points.sort_by(|a, b| a.coordinates()[axis].partial_cmp(&b.coordinates()[axis]).unwrap());
        let median = points.len() / 2;

        Some(KDTree {
            node: Some(points[median]),
            left: Box::new(Self::new_with_depth(&mut points[0..median], depth + 1)),
            right: Box::new(Self::new_with_depth(&mut points[median..], depth + 1)),
            axis: axis
        })
    }

    fn new_node(point: Point<T>, discriminator: usize) -> Box<Option<Self>> {
        Box::new(Some(KDTree {
            node: Some(point),
            left: Box::new(None),
            right: Box::new(None),
            axis: discriminator
        }))
    }

    pub fn insert(&mut self, point: Point<T>) {
        if self.node == None {
            *self = Self::new(&mut [point])
        }

        let mut cur_node = self;

        while true {
            if cur_node.node.unwrap().coordinates()[cur_node.axis] <= point.coordinates()[cur_node.axis] {
                match *cur_node.right.deref() {
                    Some(mut right) => cur_node = &mut right,
                    None => {
                        let discriminator = (cur_node.axis + 1) % point.coordinates().len();
                        cur_node.right = Self::new_node(point, discriminator);
                        return
                    }
                }
            } else {
                match *cur_node.left.deref() {
                    Some(mut left) => cur_node = &mut left,
                    None => {
                        let discriminator = (cur_node.axis + 1) % point.coordinates().len();
                        cur_node.left = Self::new_node(point, discriminator);
                        return
                    }
                }
            }
        }
    }

    pub fn remove(&mut self, point: Point<T>) -> bool {
        let node_to_remove = match self.find_node(point) {
            Some(node) => node,
            None => return false
        };

        // todo
        let parent = *node_to_remove.left.deref();
        let mut minimal_node = self.recursive_remove(node_to_remove);
        match parent {
            Some(mut node) => {
                match (*node.left.deref(), *node.right.deref(), node_to_remove) {
                    (Some(left), _, n) if left == n => node.left = Box::new(minimal_node),
                    (_, Some(right), n) if right == n => node.right = Box::new(minimal_node),
                    _ => ()
                }
            },
            None => *self = minimal_node.unwrap_or(Self::new(vec![]))
        }

        return true
    }

    fn recursive_remove(&self, node_to_remove: KDTree<T>) -> Option<Self> {
        match (*node_to_remove.left.deref(), *node_to_remove.right.deref()) {
            (None, None) => return None,
            _ => ()
        }

        let discriminator = node_to_remove.axis;
        match *node_to_remove.right.deref() {
            None => {
                node_to_remove.right = node_to_remove.left;
                node_to_remove.left = Box::new(None);
            },
            _ => ()
        }

        let (parent, minimal_node) = match self.find_minimal_node(&node_to_remove, discriminator) {
            None => return None,
            Some((_, None)) => return None,
            Some((p, Some(node))) => (p, node)
        };

        match (*parent.left.deref(), *parent.right.deref(), node_to_remove) {
            (Some(left), _, n) if left == n => parent.left = Box::new(self.recursive_remove(minimal_node)),
            (_, Some(right), n) if right == n => parent.right = Box::new(self.recursive_remove(minimal_node)),
            _ => ()
        }

        //parent = node_to_remove.parent;
        minimal_node.axis = node_to_remove.axis;
        minimal_node.right = node_to_remove.right;
        minimal_node.left = node_to_remove.left;

        Some(minimal_node)
    }

    pub fn nearest_neighbor(&mut self, point: Point<T>) -> Option<Point<T>> {
        None
    }

    // todo: should return tuple (parent, node)
    fn find_node(&self, point: Point<T>) -> Option<Self> {
        let mut found = None;

        if self.node == None {
            return found
        }

        let mut cur_node = self;

        while true {
            if cur_node.node.unwrap().coordinates()[cur_node.axis] <= point.coordinates()[cur_node.axis] {
                if cur_node.node == Some(point) {
                    found = Some(*cur_node);
                    break;
                }

                match *cur_node.right.deref() {
                    Some(right) => cur_node = &right,
                    None => ()
                }
            } else {
                match *cur_node.left.deref() {
                    Some(left) => cur_node = &left,
                    None => ()
                }
            }
        }

        found
    }

    fn find_minimal_node(&self, node_head: &KDTree<T>, discriminator: usize) -> Option<(Self, Option<Self>)> {
        let mut cur_node = (*node_head, *node_head.right.deref());
        let mut stack = vec![];
        let mut candidates = vec![];

        while true {
            match cur_node {
                (parent, Some(node)) => {
                    stack.push(cur_node);
                    cur_node = (node, *node.left.deref());
                },
                (_, None) => {
                    match stack.len() {
                        0 => break,
                        _ => {
                            cur_node = stack.pop().unwrap();
                            candidates.push(cur_node);

                            match cur_node {
                                (parent, Some(node)) => {

                                },
                                (_, None) => {

                                }
                            }
                        }
                    }
                }
            }
        }

        candidates.into_iter().min_by(|&(x_p, x_c), &(y_p, y_c)| x_c.unwrap().cmp(&y_c.unwrap()))
    }
}

impl <T: Clone + Eq + PartialEq> Ord for KDTree<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl <T: Clone + Eq + PartialEq> PartialOrd for KDTree<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.node.unwrap_or(vec![]).coordinates()[self.axis].partial_cmp(&other.node.unwrap_or(vec![]).coordinates()[other.axis])
    }
}