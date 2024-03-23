use std::mem::MaybeUninit;

/// One-way node that stores an arbitrary amount of data.
pub struct Node<T> {
	data: Vec<T>,
	len: usize,
	next: Option<Box<Node<T>>>,
}

impl<T> Default for Node<T> {
	fn default() -> Self {
		Self {
			data: Vec::default(),
			len: 0,
			next: None,
		}
	}
}

impl<T> Node<T> {
	pub fn reserve(&mut self, additional: usize) {
		self.data.reserve(additional);
	}
}

impl<T> IntoIterator for Node<T> {
	type Item = T;
	type IntoIter = NodeIter<T>;
	fn into_iter(mut self) -> Self::IntoIter {
		if self.len > self.data.len() {
			unsafe {
				// SAFETY: `len` is only incremented when the capacity is
				// initialized manually.
				self.data.set_len(self.len);
			}
		} else {
			debug_assert_eq!(self.len, self.data.len());
		}
		NodeIter {
			iter: self.data.into_iter(),
			node: self.next.map(|x| *x),
		}
	}
}

/// A view into a [`Node`], used to initialize its data in-place.
pub struct NodeWriter<'n, T> {
	data: std::slice::IterMut<'n, MaybeUninit<T>>,
	len: &'n mut usize,
	next: &'n mut Option<Box<Node<T>>>,
}

impl<'n, T> NodeWriter<'n, T> {
	pub fn new(node: &'n mut Node<T>) -> Self {
		Self {
			data: node.data.spare_capacity_mut().iter_mut(),
			len: &mut node.len,
			next: &mut node.next,
		}
	}
	
	pub fn write(&mut self, value: T) -> &'n mut T {
		let case = if let Some(case) = self.data.next() {
			case
		}
		
		 // Allocate Another Node:
		else {
			*self.next = Some(Box::new(Node {
				data: Vec::with_capacity(*self.len * 2),
				len: 0,
				next: None,
			}));
			if let Some(node) = self.next.as_mut() {
				*self = unsafe { NodeWriter {
					// SAFETY: The current `self` is being replaced - all
					// previous nodes are inaccessible and will remain in place
					// until the lifetime 'n ends.
					data: std::mem::transmute(node.data.spare_capacity_mut().iter_mut()),
					len:  std::mem::transmute(&mut node.len),
					next: std::mem::transmute(&mut node.next),
				} };
				if let Some(case) = self.data.next() {
					case
				} else {
					unreachable!()
				}
			} else {
				unreachable!()
			}
		};
		
		let value = case.write(value);
		*self.len += 1;
		
		value
	}
}

/// An iterator over the data of a [`Node`] and its sub-nodes.
pub struct NodeIter<T> {
	node: Option<Node<T>>,
	iter: std::vec::IntoIter<T>,
}

impl<T> Iterator for NodeIter<T> {
	type Item = T;
	fn next(&mut self) -> Option<Self::Item> {
		if let Some(next) = self.iter.next() {
			Some(next)
		} else if let Some(node) = self.node.take() {
			*self = node.into_iter();
			self.next()
		} else {
			None
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		let lower = self.iter.size_hint().0
			+ self.node.as_ref().map(|n| n.len).unwrap_or(0);
		(lower, None)
	}
}