use std::hash::Hash;
use std::marker::PhantomData;
use std::time::Duration;
use bevy_ecs::change_detection::{DetectChanges, Ref, Res};
use bevy_ecs::component::{Component, Tick};
use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::{Query, Resource, World};
use bevy_ecs::query::{QueryFilter, QueryIter};
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

/// Unit types that define what `PredParam::comb` iterates over.
pub trait CombKind {
	fn wrap<P: PredParam>(item: (P::Item<'_>, P::Id)) -> Option<CombCase<P>>;
	fn is_all() -> bool;
}

/// Returns all items.
pub struct CombAll;

impl CombKind for CombAll {
	fn wrap<P: PredParam>((item, id): (P::Item<'_>, P::Id)) -> Option<CombCase<P>> {
		Some(if item.is_updated() {
			CombCase::Diff(item.gimme_ref(), id)
		} else {
			CombCase::Same(item.gimme_ref(), id)
		})
	}
	fn is_all() -> bool {
		true
	}
}

/// Returns only updated items.
pub struct CombUpdated;

impl CombKind for CombUpdated {
	fn wrap<P: PredParam>((item, id): (P::Item<'_>, P::Id)) -> Option<CombCase<P>> {
		if item.is_updated() {
			Some(CombCase::Diff(item.gimme_ref(), id))
		} else {
			None
		}
	}
	fn is_all() -> bool {
		false
	}
}

/// An item & ID pair of a `PredParam`, with their updated state.
pub enum CombCase<'w, P: PredParam + ?Sized> {
	Diff(<P::Item<'w> as PredItem<'w>>::Ref<'w>, P::Id),
	Same(<P::Item<'w> as PredItem<'w>>::Ref<'w>, P::Id),
}

impl<P: PredParam> Copy for CombCase<'_, P> {}

impl<P: PredParam> Clone for CombCase<'_, P> {
	fn clone(&self) -> Self {
		*self
	}
}

impl<'w, P: PredParam> CombCase<'w, P> {
	fn id(&self) -> P::Id {
		let (Self::Diff(_, id) | Self::Same(_, id)) = self;
		*id
	}
	fn item(&self) -> <P::Item<'w> as PredItem<'w>>::Ref<'w> {
		let (Self::Diff(item, _) | Self::Same(item, _)) = self;
		*item
	}
	fn into_inner(self) -> (<P::Item<'w> as PredItem<'w>>::Ref<'w>, P::Id) {
		let (Self::Diff(item, id) | Self::Same(item, id)) = self;
		(item, id)
	}
	fn is_diff(&self) -> bool {
		match self {
			Self::Diff(..) => true,
			Self::Same(..) => false,
		}
	}
}

/// A set of [`PredItem`] values used to predict & schedule events.
pub trait PredParam {
	/// The equivalent [`bevy_ecs::system::SystemParam`].
	type Param: ReadOnlySystemParam;
	
	/// The item that `Param` iterates over.
	type Item<'w>: PredItem<'w>;
	
	/// Unique identifier of each `Item`.
	type Id: PredId;
	
	/// Creates iterators over `Param`'s items with their IDs and updated state.
	type Comb<'w, 's, K: CombKind>: IntoIterator<Item = CombCase<'w, Self>>;
	
	/// Produces `Self::Comb`.
	fn comb<'w, 's, K: CombKind>(param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::Comb<'w, 's, K>;
}

impl<T: Component, F: QueryFilter + 'static> PredParam for Query<'_, '_, &T, F> {
	type Param = Query<'static, 'static, (Ref<'static, T>, Entity), F>;
	type Item<'w> = Ref<'w, T>;
	type Id = Entity;
	type Comb<'w, 's, K: CombKind> = std::iter::FilterMap<
		QueryIter<'w, 's, (Ref<'static, T>, Entity), F>,
		fn((Ref<'w, T>, Entity)) -> Option<CombCase<'w, Self>>
	>;
	fn comb<'w, 's, K: CombKind>(param: &SystemParamItem<'w, 's, Self::Param>)
		-> Self::Comb<'w, 's, K>
	{
		param.iter_inner().filter_map(K::wrap)
	}
}

impl<R: Resource> PredParam for Res<'_, R> {
	type Param = Res<'static, R>;
	type Item<'w> = Res<'w, R>;
	type Id = ();
	type Comb<'w, 's, K: CombKind> = Option<CombCase<'w, Self>>;
	fn comb<'w, 's, K: CombKind>(param: &SystemParamItem<'w, 's, Self::Param>)
        -> Self::Comb<'w, 's, K>
	{
		K::wrap((Res::clone(param), ()))
	}
}

impl PredParam for () {
	type Param = ();
	type Item<'w> = ();
	type Id = ();
	type Comb<'w, 's, K: CombKind> = Option<CombCase<'w, Self>>;
	fn comb<'w, 's, K: CombKind>(param: &SystemParamItem<'w, 's, Self::Param>)
        -> Self::Comb<'w, 's, K>
	{
		K::wrap((*param, ()))
	}
}

impl<A: PredParam, B: PredParam> PredParam for (A, B) {
	type Param = (A::Param, B::Param);
	type Item<'w> = (A::Item<'w>, B::Item<'w>);
	type Id = (A::Id, B::Id);
	type Comb<'w, 's, K: CombKind> = PredPairComb<'w, 's, K, A, B>;
	fn comb<'w, 's, K: CombKind>(param: &SystemParamItem<'w, 's, Self::Param>)
        -> Self::Comb<'w, 's, K>
	{
		PredPairComb::new(&param.0, &param.1)
	}
}

impl<T: PredParam, const N: usize> PredParam for [T; N]
where
	T::Id: Ord
{
	type Param = T::Param;
	type Item<'w> = [T::Item<'w>; N];
	type Id = [T::Id; N];
	type Comb<'w, 's, K: CombKind> = PredArrayComb<'w, 's, K, T, N>;
	fn comb<'w, 's, K: CombKind>(param: &SystemParamItem<'w, 's, Self::Param>)
        -> Self::Comb<'w, 's, K>
	{
		PredArrayComb::new(param)
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
	state: SystemParamItem<'w, 's, P::Param>,
	node: &'p mut Node<PredStateCase<P::Id>>,
}

impl<'w, 's, 'p, P: PredParam> PredState<'w, 's, 'p, P> {
	pub(crate) fn new(
		state: SystemParamItem<'w, 's, P::Param>,
		node: &'p mut Node<PredStateCase<P::Id>>,
	) -> Self {
		Self { state, node }
	}
	
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
	type Item = <Self::IntoIter as IntoIterator>::Item;
	type IntoIter = PredCombinator<'w, 's, 'p, P>;
	fn into_iter(self) -> Self::IntoIter {
		let iter = P::comb::<CombUpdated>(&self.state).into_iter();
		self.node.reserve(4 * iter.size_hint().0.max(1));
		PredCombinator {
			iter,
			node: NodeWriter::new(self.node),
		}
	}
}

/// A scheduled case of prediction, used in [`crate::PredState`].
pub struct PredStateCase<I> {
	id: I,
	times: Option<Box<dyn Iterator<Item = (Duration, Duration)> + Send + Sync>>,
}

impl<I: PredId> PredStateCase<I> {
	fn new(id: I) -> Self {
		Self {
			id,
			times: None,
		}
	}
	
	pub(crate) fn into_parts(self)
		-> (I, Option<Box<dyn Iterator<Item = (Duration, Duration)> + Send + Sync>>)
	{
		(self.id, self.times)
	}
	
	pub fn set<T>(&mut self, times: TimeRanges<T>)
	where
		TimeRanges<T>: Iterator<Item = (Duration, Duration)> + Send + Sync + 'static
	{
		self.times = Some(Box::new(times));
	}
}

/// Types that can be used to query for a specific entity.
pub trait PredQueryData {
	type Id: PredId;
	type Output<'w>;
	unsafe fn get_inner(world: UnsafeWorldCell, id: Self::Id) -> Self::Output<'_>;
	// !!! Could take a dynamically-dispatched ID and attempt downcasting
	// manually. Return an `Option<Self::Output>` for whether it worked. This
	// would allow for query types that accept multiple IDs; support `() -> ()`.
}

impl PredQueryData for () {
	type Id = ();
	type Output<'w> = ();
	unsafe fn get_inner(_world: UnsafeWorldCell, _id: Self::Id) -> Self::Output<'_> {}
}

impl<C: Component> PredQueryData for &C {
	type Id = Entity;
	type Output<'w> = &'w C;
	unsafe fn get_inner(world: UnsafeWorldCell, id: Self::Id) -> Self::Output<'_> {
		// SAFETY: The caller should ensure that there isn't conflicting access
		// to the given entity's component.
		world.get_entity(id)
			.expect("entity should exist")
			.get::<C>()
			.expect("component should exist")
	}
}

impl<C: Component> PredQueryData for &mut C {
	type Id = Entity;
	type Output<'w> = Mut<'w, C>;
	unsafe fn get_inner(world: UnsafeWorldCell, id: Self::Id) -> Self::Output<'_> {
		// SAFETY: The caller should ensure that there isn't conflicting access
		// to the given entity's component.
		world.get_entity(id)
			.expect("entity should exist")
			.get_mut::<C>()
			.expect("component should exist")
	}
}

impl<C: Component, const N: usize> PredQueryData for [&C; N] {
	type Id = [Entity; N];
	type Output<'w> = [&'w C; N];
	unsafe fn get_inner(world: UnsafeWorldCell, id: Self::Id) -> Self::Output<'_> {
		// SAFETY: The caller should ensure that there isn't conflicting access
		// to the given entity's components.
		std::array::from_fn(|i| world.get_entity(id[i])
			.expect("entity should exist")
			.get::<C>()
			.expect("component should exist"))
	}
}

impl<C: Component, const N: usize> PredQueryData for [&mut C; N] {
	type Id = [Entity; N];
	type Output<'w> = [Mut<'w, C>; N];
	unsafe fn get_inner(world: UnsafeWorldCell, id: Self::Id) -> Self::Output<'_> {
		// SAFETY: The caller should ensure that there isn't conflicting access
		// to the given entity's components.
		std::array::from_fn(|i| world.get_entity(id[i])
			.expect("entity should exist")
			.get_mut::<C>()
			.expect("component should exist"))
	}
}

impl<A: PredQueryData, B: PredQueryData> PredQueryData for (A, B) {
	type Id = (A::Id, B::Id);
	type Output<'w> = (A::Output<'w>, B::Output<'w>);
	unsafe fn get_inner(world: UnsafeWorldCell, (a, b): Self::Id) -> Self::Output<'_> {
		(A::get_inner(world, a), B::get_inner(world, b))
	}
}

/// Prediction data fed as a parameter to an event's systems.
pub struct PredQuery<'world, 'state, P: PredQueryData> {
    world: UnsafeWorldCell<'world>,
    state: &'state P::Id,
}

impl<'w, 's, P: PredQueryData> PredQuery<'w, 's, P> {
	pub fn get_inner(self) -> P::Output<'w> {
		unsafe {
			// SAFETY: Right now this method consumes `self`. If it could be
			// called multiple times, the returned values would overlap.
			<P as PredQueryData>::get_inner(self.world, *self.state)
		}
	}
}

unsafe impl<P: PredQueryData> SystemParam for PredQuery<'_, '_, P> {
	type State = P::Id;
	type Item<'world, 'state> = PredQuery<'world, 'state, P>;
	fn init_state(world: &mut World, system_meta: &mut SystemMeta) -> Self::State {
		// !!! Check for component access overlap. This isn't safe right now.
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
	unsafe fn get_param<'world, 'state>(state: &'state mut Self::State, _system_meta: &SystemMeta, world: UnsafeWorldCell<'world>, _change_tick: Tick) -> Self::Item<'world, 'state> {
		PredQuery { world, state }
	}
}

/// Combinator for `PredParam` tuple implementation.
pub struct PredPairComb<'w, 's, K, A, B>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	a_iter: <A::Comb<'w, 's, CombAll> as IntoIterator>::IntoIter,
	b_slice: Box<[<B::Comb<'w, 's, CombAll> as IntoIterator>::Item]>,
	b_iter: <B::Comb<'w, 's, K> as IntoIterator>::IntoIter,
}

impl<'w, 's, K, A, B> PredPairComb<'w, 's, K, A, B>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	fn new(
		a_param: &SystemParamItem<'w, 's, A::Param>,
		b_param: &SystemParamItem<'w, 's, B::Param>,
	) -> Self {
		Self {
			a_iter: A::comb::<CombAll>(a_param).into_iter(),
			b_slice: B::comb::<CombAll>(b_param).into_iter().collect(),
			b_iter: B::comb::<K>(b_param).into_iter(),
		}
	}
}

impl<'w, 's, K, A, B> IntoIterator for PredPairComb<'w, 's, K, A, B>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	type Item = CombCase<'w, (A, B)>;
	type IntoIter = PredPairIter<'w, 's, K, A, B>;
	fn into_iter(self) -> Self::IntoIter {
		let Self { a_iter, b_slice, b_iter } = self;
		let ((_, Some(size)) | (size, None)) = a_iter.size_hint(); // !!! Maybe don't use upper bound
		let a_vec = Vec::with_capacity(size);
		PredPairIter::primary_next(a_iter, a_vec, b_slice, b_iter)
	}
}

/// Iterator for 2-tuple [`PredParam`] types.
pub enum PredPairIter<'w, 's, K, A, B>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	Empty,
	Primary {
		a_iter: <A::Comb<'w, 's, CombAll> as IntoIterator>::IntoIter,
		a_curr: <A::Comb<'w, 's, CombUpdated> as IntoIterator>::Item,
		a_vec: Vec<<A::Comb<'w, 's, CombUpdated> as IntoIterator>::Item>,
		b_slice: Box<[<B::Comb<'w, 's, CombUpdated> as IntoIterator>::Item]>,
		b_index: usize,
		b_iter: <B::Comb<'w, 's, K> as IntoIterator>::IntoIter,
	},
	Secondary {
		b_iter: <B::Comb<'w, 's, K> as IntoIterator>::IntoIter,
		b_curr: <B::Comb<'w, 's, CombUpdated> as IntoIterator>::Item,
		a_slice: Box<[<A::Comb<'w, 's, CombUpdated> as IntoIterator>::Item]>,
		a_index: usize,
	},
}

impl<'w, 's, A, B, K> PredPairIter<'w, 's, K, A, B>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	fn primary_next(
		mut a_iter: <A::Comb<'w, 's, CombAll> as IntoIterator>::IntoIter,
		mut a_vec: Vec<<A::Comb<'w, 's, CombUpdated> as IntoIterator>::Item>,
		b_slice: Box<[<B::Comb<'w, 's, CombUpdated> as IntoIterator>::Item]>,
		mut b_iter: <B::Comb<'w, 's, K> as IntoIterator>::IntoIter,
	) -> Self {
		while let Some(a_curr) = a_iter.next() {
			if a_curr.is_diff() {
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
		mut b_iter: <B::Comb<'w, 's, K> as IntoIterator>::IntoIter,
		a_slice: Box<[<A::Comb<'w, 's, CombUpdated> as IntoIterator>::Item]>,
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

impl<'w, 's, K, A, B> Iterator for PredPairIter<'w, 's, K, A, B>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	type Item = CombCase<'w, (A, B)>;
	fn next(&mut self) -> Option<Self::Item> {
		// !!! Put A/B in order of ascending size to reduce redundancy.
		match std::mem::replace(self, Self::Empty) {
			Self::Empty => None,
			
			 // (Updated A, All B): 
			Self::Primary {
				a_iter,
				a_curr,
				a_vec,
				b_slice,
				b_index,
				b_iter,
			} => {
				if let Some(b_curr) = b_slice.get(b_index).copied() {
					*self = Self::Primary {
						a_iter,
						a_curr,
						a_vec,
						b_slice,
						b_index: b_index + 1,
						b_iter,
					};
					let (a, a_id) = a_curr.into_inner();
					let (b, b_id) = b_curr.into_inner();
					return Some(if a_curr.is_diff() || b_curr.is_diff() {
						CombCase::Diff((a, b), (a_id, b_id))
					} else {
						CombCase::Same((a, b), (a_id, b_id))
					})
				}
				*self = Self::primary_next(a_iter, a_vec, b_slice, b_iter);
				self.next()
			},
			
			 // (Updated B, Non-updated A):
			Self::Secondary {
				b_iter,
				b_curr,
				a_slice,
				a_index,
			} => {
				if let Some(a_curr) = a_slice.get(a_index).copied() {
					*self = Self::Secondary {
						b_iter,
						b_curr,
						a_slice,
						a_index: a_index + 1,
					};
					let (a, a_id) = a_curr.into_inner();
					let (b, b_id) = b_curr.into_inner();
					return Some(if a_curr.is_diff() || b_curr.is_diff() {
						CombCase::Diff((a, b), (a_id, b_id))
					} else {
						CombCase::Same((a, b), (a_id, b_id))
					})
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

/// Combinator for `PredParam` array implementation.
pub struct PredArrayComb<'w, 's, K, P, const N: usize>
where
	K: CombKind,
	P: PredParam,
{
	slice: Box<[<P::Comb<'w, 's, K> as IntoIterator>::Item]>,
}

impl<'w, 's, K, P, const N: usize> PredArrayComb<'w, 's, K, P, N>
where
	K: CombKind,
	P: PredParam,
	P::Id: Ord,
{
	fn new(param: &SystemParamItem<'w, 's, P::Param>) -> Self {
		let mut vec = P::comb::<CombAll>(param).into_iter().collect::<Vec<_>>();
		vec.sort_unstable_by_key(|x| x.id());
		Self {
			slice: vec.into_boxed_slice()
		}
	}
}

impl<'w, 's, K, P, const N: usize> IntoIterator for PredArrayComb<'w, 's, K, P, N>
where
	K: CombKind,
	P: PredParam,
	P::Id: Ord,
{
	type Item = CombCase<'w, [P; N]>;
	type IntoIter = PredArrayIter<'w, 's, K, P, N>;
	fn into_iter(self) -> Self::IntoIter {
		let mut iter = PredArrayIter {
			slice: self.slice,
			index: [0; N],
			is_first: true,
			kind: PhantomData,
		};
		iter.step_main();
		iter
	}
}

/// Iterator for array of [`PredParam`] type.
pub struct PredArrayIter<'w, 's, K, T, const N: usize>
where
	K: CombKind,
	T: PredParam,
{
	slice: Box<[<T::Comb<'w, 's, CombAll> as IntoIterator>::Item]>,
	index: [usize; N],
	is_first: bool,
	kind: PhantomData<K>,
}

impl<'w, 's, K, T, const N: usize> PredArrayIter<'w, 's, K, T, N>
where
	K: CombKind,
	T: PredParam,
	T::Id: Ord,
{
	fn step_main(&mut self) {
		//! Moves the main index to the next updated item.
		while let Some(x) = self.slice.get(self.index[N-1]) {
			if x.is_diff() && self.step_sub(N-1) {
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
			while index <= self.index[N-1] && (K::is_all() || self.slice[index].is_diff()) {
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

impl<'w, 's, K, T, const N: usize> Iterator for PredArrayIter<'w, 's, K, T, N>
where
	K: CombKind,
	T: PredParam,
	T::Id: Ord,
{
	type Item = CombCase<'w, [T; N]>;
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
			while index <= self.index[N-1] && (K::is_all() || self.slice[index].is_diff()) {
				index += 1;
			}
			if index >= self.slice.len() {
				self.index[i+1] += 1;
				continue
			}
			self.index[i] = index;
			
			if self.step_sub(i) {
				let (mut refs, mut ids) = (
					self.index.map(|i| self.slice[i].item()),
					self.index.map(|i| self.slice[i].id()),
				);
				let mut last = N-1;
				while last != 0 && ids[last] > ids[last - 1] {
					ids .swap(last, last - 1);
					refs.swap(last, last - 1);
					last -= 1;
				}
				self.index[0] += 1;
				return Some(if self.slice.iter().any(CombCase::is_diff) {
					CombCase::Diff(refs, ids)
				} else {
					CombCase::Same(refs, ids)
				})
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
	iter: <P::Comb<'w, 's, CombUpdated> as IntoIterator>::IntoIter,
	node: NodeWriter<'p, PredStateCase<P::Id>>,
}

impl<'w, 's, 'p, P: PredParam> Iterator for PredCombinator<'w, 's, 'p, P> {
	type Item = (
		&'p mut PredStateCase<P::Id>,
		<P::Item<'w> as PredItem<'w>>::Ref<'w>
	);
	fn next(&mut self) -> Option<Self::Item> {
		if let Some(case) = self.iter.next() {
			let (item, id) = case.into_inner();
			Some((
				self.node.write(PredStateCase::new(id)),
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