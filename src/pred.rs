use std::hash::Hash;
use std::time::Duration;
use bevy_ecs::change_detection::{DetectChanges, Ref, Res};
use bevy_ecs::component::{Component, Tick};
use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::{Query, Resource, World};
use bevy_ecs::query::{QueryData, QueryEntityError, QueryFilter, QueryItem, QueryIter, ROQueryItem};
use bevy_ecs::system::{ReadOnlySystemParam, SystemMeta, SystemParam, SystemParamItem};
use bevy_ecs::world::{Mut, unsafe_world_cell::UnsafeWorldCell};
use chime::time::TimeRanges;
use crate::node::*;

/// Resource for passing an event's unique ID to its system parameters. 
#[derive(Resource)]
pub(crate) struct PredSystemId(pub Box<dyn std::any::Any + Send + Sync>);

/// A hashable unique identifier for a case of prediction.
pub trait PredId:
	Copy + Clone + Eq + Hash + std::fmt::Debug + Send + Sync + 'static
{}

impl<T> PredId for T
where
	T: Copy + Clone + Eq + Hash + std::fmt::Debug + Send + Sync + 'static
{}

/// A set of [`PredItem`] values used to predict & schedule events.
pub trait PredParam {
	/// The equivalent [`bevy_ecs::system::SystemParam`].
	type Param: ReadOnlySystemParam;
	
	/// The item that `Param` iterates over.
	type Item<'w>: PredItem<'w>;
	
	/// Unique identifier of each `Item`.
	type Id: PredId;
	
	/// Iterator over `Param`'s items alongside their `is_updated` state.
	type Iterator<'w, 's>:
		Iterator<Item = ((<Self::Item<'w> as PredItem<'w>>::Ref<'w>, Self::Id), bool)>;
	
	/// Iterator over `Param`'s updated items.
	type UpdatedIterator<'w, 's>:
		Iterator<Item = (<Self::Item<'w> as PredItem<'w>>::Ref<'w>, Self::Id)>;
	
	/// Produces `Self::Iterator`.
	fn gimme_iter<'w, 's>(param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::Iterator<'w, 's>;
	
	/// Produces `Self::UpdatedIterator`.
	fn updated_iter<'w, 's>(param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::UpdatedIterator<'w, 's>;
}

impl<T: Component> PredParam for Query<'_, '_, (Ref<'_, T>, Entity)> {
	type Param = Query<'static, 'static, (Ref<'static, T>, Entity)>;
	type Item<'w> = Ref<'w, T>;
	type Id = Entity;
	type Iterator<'w, 's> = std::iter::Map<
		QueryIter<'w, 's, (Ref<'static, T>, Entity), ()>,
		fn((Ref<'w, T>, Entity)) -> ((&'w T, Entity), bool)
	>;
	type UpdatedIterator<'w, 's> = std::iter::FilterMap<
		QueryIter<'w, 's, (Ref<'static, T>, Entity), ()>,
		fn((Ref<'w, T>, Entity)) -> Option<(&'w T, Entity)>
	>;
	fn gimme_iter<'w, 's>(param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::Iterator<'w, 's>
	{
		param.iter_inner().map(|(item, id)| {
			let is_updated = item.is_updated();
			((item.gimme_ref(), id), is_updated)
		})
	}
	fn updated_iter<'w, 's>(param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::UpdatedIterator<'w, 's>
	{
		param.iter_inner().filter_map(|(item, id)| {
			if item.is_updated() {
				Some((item.gimme_ref(), id))
			} else {
				None
			}
		})
	}
}

impl<R: Resource> PredParam for Res<'_, R> {
	type Param = Res<'static, R>;
	type Item<'w> = Res<'w, R>;
	type Id = ();
	type Iterator<'w, 's> = std::iter::Once<((<Self::Item<'w> as PredItem<'w>>::Ref<'w>, Self::Id), bool)>;
	type UpdatedIterator<'w, 's> = std::option::IntoIter<(<Self::Item<'w> as PredItem<'w>>::Ref<'w>, Self::Id)>;
	fn gimme_iter<'w, 's>(param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::Iterator<'w, 's>
	{
		std::iter::once((
			(Res::clone(param).gimme_ref(), ()),
			param.is_updated()
		))
	}
	fn updated_iter<'w, 's>(param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::UpdatedIterator<'w, 's>
	{
		if param.is_updated() {
			Some((Res::clone(param).gimme_ref(), ()))
		} else {
			None
		}.into_iter()
	}
}

impl PredParam for () {
	type Param = ();
	type Item<'w> = ();
	type Id = ();
	type Iterator<'w, 's> = std::iter::Once<(((), ()), bool)>;
	type UpdatedIterator<'w, 's> = std::iter::Once<((), ())>;
	fn gimme_iter<'w, 's>(_param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::Iterator<'w, 's>
	{
		std::iter::once((((), ()), true))
	}
	fn updated_iter<'w, 's>(_param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::UpdatedIterator<'w, 's>
	{
		std::iter::once(((), ()))
	}
}

impl<A: PredParam, B: PredParam> PredParam for (A, B) {
	type Param = (A::Param, B::Param);
	type Item<'w> = (A::Item<'w>, B::Item<'w>);
	type Id = (A::Id, B::Id);
	type Iterator<'w, 's> = std::vec::IntoIter<((<Self::Item<'w> as PredItem<'w>>::Ref<'w>, Self::Id), bool)>;
	type UpdatedIterator<'w, 's> = PredGroupIter<'w, 's, A, B>;
	fn gimme_iter<'w, 's>(param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::Iterator<'w, 's> 
	{
		// !!! Change this later.
		let mut vec = Vec::new();
		for ((a, a_id), a_is_updated) in A::gimme_iter(&param.0) {
			for ((b, b_id), b_is_updated) in B::gimme_iter(&param.1) {
				vec.push((
					((a, b), (a_id, b_id)),
					a_is_updated || b_is_updated,
				));
			}
		}
		vec.into_iter()
	}
	fn updated_iter<'w, 's>(param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::UpdatedIterator<'w, 's>
	{
		PredGroupIter::new(&param.0, &param.1)
	}
}

impl<T: PredParam, const N: usize> PredParam for [T; N]
where
	T::Id: Ord
{
	type Param = T::Param;
	type Item<'w> = [T::Item<'w>; N];
	type Id = [T::Id; N];
	type Iterator<'w, 's> = std::vec::IntoIter<((<Self::Item<'w> as PredItem<'w>>::Ref<'w>, Self::Id), bool)>;
	type UpdatedIterator<'w, 's> = PredArrayIter<'w, 's, T, N>;
	fn gimme_iter<'w, 's>(_param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::Iterator<'w, 's>
	{
		todo!()
	}
	fn updated_iter<'w, 's>(param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::UpdatedIterator<'w, 's>
	{
		PredArrayIter::new(param)
	}
}

/// A case of prediction.
pub trait PredItem<'w> {
	type Ref<'i>: Copy/* + std::ops::Deref<Target=Self::Inner>*/;
	type Inner: 'w;
	
	/// Needed because `bevy::ecs::world::Ref` can't be cloned/copied.
	fn gimme_ref(self) -> Self::Ref<'w>;
	
	/// Whether this item is in need of a prediction update.
	fn is_updated(&self) -> bool;
}

impl<'w, T: 'static> PredItem<'w> for Ref<'w, T> {
	type Ref<'i> = &'i Self::Inner;
	type Inner = T;
	fn gimme_ref(self) -> Self::Ref<'w> {
		Ref::into_inner(self)
	}
	fn is_updated(&self) -> bool {
		DetectChanges::is_changed(self)
	}
}

impl<'w, R: Resource> PredItem<'w> for Res<'w, R> {
	type Ref<'i> = &'i Self::Inner;
	type Inner = R;
	fn gimme_ref(self) -> Self::Ref<'w> {
		Res::into_inner(self)
	}
	fn is_updated(&self) -> bool {
		DetectChanges::is_changed(self)
	}
}

impl<'w> PredItem<'w> for () {
	type Ref<'i> = ();
	type Inner = ();
	fn gimme_ref(self) -> Self::Ref<'w> {}
	fn is_updated(&self) -> bool {
		true
	}
}

impl<'w, A, B> PredItem<'w> for (A, B)
where
	A: PredItem<'w>,
	B: PredItem<'w>,
{
	type Ref<'i> = (A::Ref<'i>, B::Ref<'i>);
	type Inner = (A::Inner, B::Inner);
	fn gimme_ref(self) -> Self::Ref<'w> {
		(self.0.gimme_ref(), self.1.gimme_ref())
	}
	fn is_updated(&self) -> bool {
		self.0.is_updated() || self.1.is_updated()
	}
}

impl<'w, T, const N: usize> PredItem<'w> for [T; N]
where
	T: PredItem<'w>
{
	type Ref<'i> = [T::Ref<'i>; N];
	type Inner = [T::Inner; N];
	fn gimme_ref(self) -> Self::Ref<'w> {
		self.map(|x| x.gimme_ref())
	}
	fn is_updated(&self) -> bool {
		self.iter().any(T::is_updated)
	}
}

/// Collects predictions from "when" systems for later compilation.
pub struct PredState<'w, 's, 'p, P: PredParam = ()> {
	pub(crate) state: SystemParamItem<'w, 's, P::Param>,
	pub(crate) node: &'p mut Node<PredStateCase<P::Id>>,
}

impl<P: PredParam> PredState<'_, '_, '_, P> {
	/// Sets all updated cases to the given times.
	pub fn set<I>(self, times: TimeRanges<I>)
	where
		TimeRanges<I>: Iterator<Item = (Duration, Duration)> + Clone + Send + Sync + 'static
	{
		let mut iter = self.into_iter();
		if let Some((first, _)) = iter.next() {
			for (case, _) in iter {
				case.set(times.clone());
			}
			first.set(times);
		}
	}
}

impl<'w, 's, 'p, P: PredParam> IntoIterator for PredState<'w, 's, 'p, P> {
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = PredCombinator<'w, 's, 'p, P>;
	fn into_iter(self) -> Self::IntoIter {
		let inner = P::updated_iter(&self.state);
		self.node.reserve(4 * inner.size_hint().0.max(1));
		PredCombinator {
			iter: inner,
			node: NodeWriter::new(self.node),
		}
	}
}

/// A scheduled case of prediction, used in [`crate::PredState`].
pub struct PredStateCase<P>(
	pub(crate) Option<Box<dyn Iterator<Item = (Duration, Duration)> + Send + Sync>>,
	pub(crate) P,
);

impl<P: PredId> PredStateCase<P> {
	pub fn set<I>(&mut self, times: TimeRanges<I>)
	where
		TimeRanges<I>: Iterator<Item = (Duration, Duration)> + Send + Sync + 'static
	{
		self.0 = Some(Box::new(times));
	}
}

/// Types that can be used to query for a specific entity.
pub unsafe trait PredQueryData {
	type Id: PredId;
	type Output<'w>;
	fn get_inner(world: UnsafeWorldCell, id: Self::Id) -> Self::Output<'_>;
}

unsafe impl PredQueryData for () {
	type Id = ();
	type Output<'w> = ();
	fn get_inner(_world: UnsafeWorldCell, _id: Self::Id) -> Self::Output<'_> {}
}

unsafe impl<C: Component> PredQueryData for &C {
	type Id = Entity;
	type Output<'w> = &'w C;
	fn get_inner(world: UnsafeWorldCell, id: Self::Id) -> Self::Output<'_> {
		unsafe {
			// SAFETY: !!!
			world.get_entity(id)
				.expect("entity should exist")
				.get::<C>()
				.expect("component should exist")
		}
	}
}

unsafe impl<C: Component> PredQueryData for &mut C {
	type Id = Entity;
	type Output<'w> = Mut<'w, C>;
	fn get_inner(world: UnsafeWorldCell, id: Self::Id) -> Self::Output<'_> {
		unsafe {
			// SAFETY: !!!
			world.get_entity(id)
				.expect("entity should exist")
				.get_mut::<C>()
				.expect("component should exist")
		}
	}
}

unsafe impl<C: Component, const N: usize> PredQueryData for [&C; N] {
	type Id = [Entity; N];
	type Output<'w> = [&'w C; N];
	fn get_inner(world: UnsafeWorldCell, id: Self::Id) -> Self::Output<'_> {
		std::array::from_fn(|i| unsafe {
			// SAFETY: !!! Not really safe yet.
			world.get_entity(id[i])
				.expect("entity should exist")
				.get::<C>()
				.expect("component should exist")
		})
	}
}

unsafe impl<C: Component, const N: usize> PredQueryData for [&mut C; N] {
	type Id = [Entity; N];
	type Output<'w> = [Mut<'w, C>; N];
	fn get_inner(world: UnsafeWorldCell, id: Self::Id) -> Self::Output<'_> {
		std::array::from_fn(|i| unsafe {
			// SAFETY: !!! Not really safe yet.
			world.get_entity(id[i])
				.expect("entity should exist")
				.get_mut::<C>()
				.expect("component should exist")
		})
	}
}

unsafe impl<A: PredQueryData, B: PredQueryData> PredQueryData for (A, B) {
	type Id = (A::Id, B::Id);
	type Output<'w> = (A::Output<'w>, B::Output<'w>);
	fn get_inner(world: UnsafeWorldCell, (a, b): Self::Id) -> Self::Output<'_> {
		(A::get_inner(world, a), B::get_inner(world, b))
	}
}

/// Prediction data fed as a parameter to an event's systems.
pub struct PredInput<'world, 'state, P: PredQueryData> {
    world: UnsafeWorldCell<'world>,
    state: &'state P::Id,
}

impl<'w, 's, P: PredQueryData> PredInput<'w, 's, P> {
	pub fn get_inner(self) -> P::Output<'w> {
		<P as PredQueryData>::get_inner(self.world, *self.state)
	}
}

unsafe impl<P: PredQueryData> SystemParam for PredInput<'_, '_, P> {
	type State = P::Id;
	type Item<'world, 'state> = PredInput<'world, 'state, P>;
	fn init_state(world: &mut World, system_meta: &mut SystemMeta) -> Self::State {
		// !!! Check for component access overlap.
		if let Some(PredSystemId(id)) = world.get_resource::<PredSystemId>() {
			if let Some(id) = id.downcast_ref::<P::Id>() {
				*id
			} else {
				panic!("!!! parameter is for wrong ID type");
			}
		} else {
			panic!("!!! {:?} is not a Chime event system, it can't use this parameter type", system_meta.name());
		}
	}
	// fn new_archetype(_state: &mut Self::State, _archetype: &Archetype, _system_meta: &mut SystemMeta) {
	// 	todo!()
	// }
	unsafe fn get_param<'world, 'state>(state: &'state mut Self::State, system_meta: &SystemMeta, world: UnsafeWorldCell<'world>, change_tick: Tick) -> Self::Item<'world, 'state> {
		PredInput { world, state }
	}
}

/// Iterator for 2-tuple [`PredParam`] types.
pub enum PredGroupIter<'w, 's, A, B>
where
	A: PredParam,
	B: PredParam,
{
	Empty,
	Primary {
		a_iter: A::Iterator<'w, 's>,
		a_curr: <A::UpdatedIterator<'w, 's> as Iterator>::Item,
		a_vec: Vec<<A::UpdatedIterator<'w, 's> as Iterator>::Item>,
		b_slice: Box<[<B::UpdatedIterator<'w, 's> as Iterator>::Item]>,
		b_index: usize,
		b_iter: B::UpdatedIterator<'w, 's>,
	},
	Secondary {
		b_iter: B::UpdatedIterator<'w, 's>,
		b_curr: <B::UpdatedIterator<'w, 's> as Iterator>::Item,
		a_slice: Box<[<A::UpdatedIterator<'w, 's> as Iterator>::Item]>,
		a_index: usize,
	},
}

impl<'w, 's, A, B> PredGroupIter<'w, 's, A, B>
where
	A: PredParam,
	B: PredParam,
{
	pub(crate) fn new(
		a_param: &SystemParamItem<'w, 's, A::Param>,
		b_param: &SystemParamItem<'w, 's, B::Param>,
	) -> Self {
		let a_iter = A::gimme_iter(a_param);
		let ((_, Some(size)) | (size, None)) = a_iter.size_hint();
		let a_vec = Vec::with_capacity(size);
		let b_slice = B::gimme_iter(b_param).map(|(x, _)| x).collect();
		let b_iter = B::updated_iter(b_param);
		Self::primary_next(a_iter, a_vec, b_slice, b_iter)
	}
	
	fn primary_next(
		mut a_iter: A::Iterator<'w, 's>,
		mut a_vec: Vec<<A::UpdatedIterator<'w, 's> as Iterator>::Item>,
		b_slice: Box<[<B::UpdatedIterator<'w, 's> as Iterator>::Item]>,
		mut b_iter: B::UpdatedIterator<'w, 's>,
	) -> Self {
		while let Some((a_curr, a_is_updated)) = a_iter.next() {
			if a_is_updated {
				return Self::Primary {
					a_iter,
					a_curr,
					a_vec,
					b_slice,
					b_index: 0,
					b_iter,
				}
			}
			a_vec.push(a_curr);
		}
		
		 // Switch to Secondary Iteration:
		if let Some(b_curr) = b_iter.next() {
			return Self::Secondary {
				b_iter,
				b_curr,
				a_slice: a_vec.into_boxed_slice(),
				a_index: 0,
			}
		}
		
		Self::Empty
	}
	
	fn secondary_next(
		mut b_iter: B::UpdatedIterator<'w, 's>,
		a_slice: Box<[<A::UpdatedIterator<'w, 's> as Iterator>::Item]>,
	) -> Self {
		if let Some(b_curr) = b_iter.next() {
			return Self::Secondary {
				b_iter,
				b_curr,
				a_slice,
				a_index: 0,
			}
		}
		Self::Empty
	}
}

impl<'w, 's, A, B> Iterator for PredGroupIter<'w, 's, A, B>
where
	A: PredParam,
	B: PredParam,
{
	type Item = (
		<<(A, B) as PredParam>::Item<'w> as PredItem<'w>>::Ref<'w>,
		<(A, B) as PredParam>::Id
	);
	fn next(&mut self) -> Option<Self::Item> {
		// !!! Put A/B in order of ascending size to reduce redundancy.
		match std::mem::replace(self, Self::Empty) {
			Self::Empty => None,
			
			 // (Updated A, All B): 
			Self::Primary {
				a_iter,
				a_curr: a_curr @ (a, a_id),
				a_vec,
				b_slice,
				b_index,
				b_iter,
			} => {
				if let Some((b, b_id)) = b_slice.get(b_index).copied() {
					*self = Self::Primary {
						a_iter,
						a_curr,
						a_vec,
						b_slice,
						b_index: b_index + 1,
						b_iter,
					};
					return Some(((a, b), (a_id, b_id)))
				}
				*self = Self::primary_next(a_iter, a_vec, b_slice, b_iter);
				self.next()
			},
			
			 // (Updated B, Non-updated A):
			Self::Secondary {
				b_iter,
				b_curr: b_curr @ (b, b_id),
				a_slice,
				a_index,
			} => {
				if let Some((a, a_id)) = a_slice.get(a_index).copied() {
					*self = Self::Secondary {
						b_iter,
						b_curr,
						a_slice,
						a_index: a_index + 1,
					};
					return Some(((a, b), (a_id, b_id)))
				}
				*self = Self::secondary_next(b_iter, a_slice);
				self.next()
			},
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		match self {
			Self::Empty => (0, Some(0)),
			Self::Primary { a_iter, a_vec, b_slice, b_index, .. } => {
				let min = b_slice.len() - b_index;
				let a_max = a_iter.size_hint().1;
				(
					min,
					a_max.map(|x| min + ((x + a_vec.len()) * b_slice.len()))
				)
			},
			Self::Secondary { b_iter, a_slice, a_index, .. } => {
				let min = a_slice.len() - a_index;
				let b_max = b_iter.size_hint().1;
				(
					min,
					b_max.map(|x| min + (x * a_slice.len()))
				)
			},
		}
	}
}

/// Iterator for array of [`PredParam`] type.
pub struct PredArrayIter<'w, 's, T: PredParam, const N: usize> {
	slice: Box<[<T::Iterator<'w, 's> as Iterator>::Item]>,
	index: [usize; N],
	is_first: bool,
}

impl<'w, 's, T: PredParam, const N: usize> PredArrayIter<'w, 's, T, N>
where
	T::Id: Ord,
{
	pub(crate) fn new(param: &SystemParamItem<'w, 's, T::Param>) -> Self {
		let mut vec = T::gimme_iter(param).collect::<Vec<_>>();
		vec.sort_unstable_by_key(|((_, id), _)| *id);
		let mut iter = Self {
			slice: vec.into_boxed_slice(),
			index: [0; N],
			is_first: true,
		};
		iter.step_main();
		iter
	}
	
	fn step_main(&mut self) {
		//! Moves the main index to the next updated item.
		while let Some(&(_, is_updated)) = self.slice.get(self.index[N-1]) {
			if is_updated && self.step_sub(N-1) {
				break
			}
			self.index[N-1] += 1;
		}
	}
	
	fn step_sub(&mut self, start: usize) -> bool {
		//! Initializes and moves the sub-indices to the next updated item.
		//! Returns false if an index exceeds the slice's length.
		let mut index = if start == N-1 {
			0
		} else {
			self.index[start] + 1
		};
		for i in (0..start).rev() {
			while index <= self.index[N-1] && self.slice[index].1 {
				index += 1;
			}
			if index >= self.slice.len() {
				return false
			}
			self.index[i] = index;
			index += 1;
		}
		true
	}
}

impl<'w, 's, T, const N: usize> Iterator for PredArrayIter<'w, 's, T, N>
where
	T: PredParam,
	T::Id: Ord,
{
	type Item = (
		<<[T; N] as PredParam>::Item<'w> as PredItem<'w>>::Ref<'w>,
		<[T; N] as PredParam>::Id
	);
	fn next(&mut self) -> Option<Self::Item> {
		// !!! Might be faster:
		// f(slice, layer):
		//   if layer == top_layer:
		//     for (index, item) in updated_slice:
		//       if index >= slice.starting_point:
		//         yield item
		//   else:
		//     for (index, item) in slice:
		//       if item.is_updated():
		//         yield all combinations of this item and the items after it
		//       else:
		//         yield item + f(slice[index+1..], layer+1)
		self.is_first = false; // Temporary `size_hint` initial lower bound.
		if self.index[N-1] >= self.slice.len() {
			return None
		}
		for i in 0..N-1 {
			let mut index = self.index[i];
			while index <= self.index[N-1] && self.slice[index].1 {
				index += 1;
			}
			if index >= self.slice.len() {
				self.index[i+1] += 1;
				continue
			}
			self.index[i] = index;
			
			if self.step_sub(i) {
				let (mut refs, mut ids) = (
					self.index.map(|i| self.slice[i].0.0), // Item::Ref
					self.index.map(|i| self.slice[i].0.1), // Id
				);
				let mut last = N-1;
				while last != 0 && ids[last] > ids[last - 1] {
					ids .swap(last, last - 1);
					refs.swap(last, last - 1);
					last -= 1;
				}
				self.index[0] += 1;
				return Some((refs, ids))
			}
			
			self.index[N-1] += 1;
			break
		}
		self.step_main();
		self.next()
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		if self.slice.get(self.index[N-1]).is_none() {
			(0, Some(0))
		} else {
			let len = self.slice.len();
			let lower = (1+len-N..len).product::<usize>()
				/ (1..N).product::<usize>(); // (len-1) choose (N-1)
			let upper = lower * len / N; // len choose N
			if self.is_first {
				(lower, Some(upper))
			} else {
				(1, Some(upper)) // !!! Improve lower bound estimation later.
			}
		}
	}
}

/// Produces all case combinations in need of a new prediction, alongside a
/// [`PredStateCase`] for scheduling.
pub struct PredCombinator<'w, 's, 'p, P: PredParam> {
	iter: P::UpdatedIterator<'w, 's>,
	node: NodeWriter<'p, PredStateCase<P::Id>>,
}

impl<'w, 's, 'p, P: PredParam> Iterator for PredCombinator<'w, 's, 'p, P> {
	type Item = (
		&'p mut PredStateCase<P::Id>,
		<P::Item<'w> as PredItem<'w>>::Ref<'w>
	);
	fn next(&mut self) -> Option<Self::Item> {
		if let Some((item, id)) = self.iter.next() {
			Some((
				self.node.write(PredStateCase(None, id)),
				item
			))
		} else {
			None
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.iter.size_hint()
	}
}