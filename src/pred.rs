use std::hash::Hash;
use std::marker::PhantomData;
use std::time::Duration;
use bevy_ecs::change_detection::{DetectChanges, Ref, Res};
use bevy_ecs::component::{Component, Tick};
use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::{Query, Resource, World};
use bevy_ecs::query::{ArchetypeFilter, QueryIter};
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
/// 
/// ```text
///         | ::Pal  | ::Inv   | ::Inv::Pal::Inv
///    None | None   | All     | None
///     All | All    | None    | All
/// Updated | All    | Static  | Updated
///  Static | Static | Updated | None
/// ```
pub trait CombKind {
	type Pal: CombKind;
	type Inv: CombKind;
	
	const HAS_DIFF: bool;
	const HAS_SAME: bool;
	
	fn wrap<'w, T, I>((item, id): (T, I)) -> Option<CombCase<'w, T, I>>
	where
		T: PredItem<'w>,
		I: PredId,
	{
		if Self::HAS_DIFF || Self::HAS_SAME {
			if T::is_updated(&item) {
				if Self::HAS_DIFF {
					return Some(CombCase::Diff(T::into_ref(item), id))
				}
			} else if Self::HAS_SAME {
				return Some(CombCase::Same(T::into_ref(item), id))
			}
		}
		None
	}
	
	fn is_all() -> bool {
		Self::HAS_DIFF && Self::HAS_SAME
	}
}

/// No combinations.
pub struct CombNone;

impl CombKind for CombNone {
	type Pal = CombNone;
	type Inv = CombAll;
	const HAS_DIFF: bool = false;
	const HAS_SAME: bool = false;
}

/// All combinations.
pub struct CombAll;

impl CombKind for CombAll {
	type Pal = CombAll;
	type Inv = CombNone;
	const HAS_DIFF: bool = true;
	const HAS_SAME: bool = true;
}

/// Combinations where either item updated.
pub struct CombUpdated;

impl CombKind for CombUpdated {
	type Pal = CombAll;
	type Inv = CombStatic;
	const HAS_DIFF: bool = true;
	const HAS_SAME: bool = false;
}

/// Combinations where neither item updated.
pub struct CombStatic;

impl CombKind for CombStatic {
	type Pal = CombStatic;
	type Inv = CombUpdated;
	const HAS_DIFF: bool = false;
	const HAS_SAME: bool = true;
}

/// An item & ID pair of a `PredParam`, with their updated state.
pub enum CombCase<'w, P: PredItem<'w>, I: PredId> {
	Diff(P::Ref<'w>, I),
	Same(P::Ref<'w>, I),
}

impl<'w, P: PredItem<'w>, I: PredId> Copy for CombCase<'w, P, I> {}

impl<'w, P: PredItem<'w>, I: PredId> Clone for CombCase<'w, P, I> {
	fn clone(&self) -> Self {
		*self
	}
}

impl<'w, P: PredItem<'w>, I: PredId> CombCase<'w, P, I> {
	fn id(&self) -> I {
		let (Self::Diff(_, id) | Self::Same(_, id)) = self;
		*id
	}
	fn item(&self) -> P::Ref<'w> {
		let (Self::Diff(item, _) | Self::Same(item, _)) = self;
		*item
	}
	fn into_inner(self) -> (P::Ref<'w>, I) {
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
	type Param: ReadOnlySystemParam + 'static;
	
	/// The item that `Param` iterates over.
	type Item<'w>: PredItem<'w>;
	
	/// Unique identifier of each `Item`.
	type Id: PredId;
	
	/// Creates iterators over `Param`'s items with their IDs and updated state.
	type Comb<'w, K: CombKind>:
		IntoIterator<Item = CombCase<'w, Self::Item<'w>, Self::Id>>
		+ Clone;
	
	/// Produces `Self::Comb`.
	fn comb<'w, K: CombKind>(param: &'w SystemParamItem<Self::Param>)
		-> Self::Comb<'w, K>;
}

impl<T: Component, F: ArchetypeFilter + 'static> PredParam for Query<'_, '_, &T, F> {
	type Param = Query<'static, 'static, (Ref<'static, T>, Entity), F>;
	type Item<'w> = Ref<'w, T>;
	type Id = Entity;
	type Comb<'w, K: CombKind> = QueryComb<'w, K, T, F>;
	fn comb<'w, K: CombKind>(param: &'w SystemParamItem<Self::Param>)
		-> Self::Comb<'w, K>
	{
		QueryComb {
			inner: param,
			kind: PhantomData,
		}
	}
}

impl<R: Resource> PredParam for Res<'_, R> {
	type Param = Res<'static, R>;
	type Item<'w> = Res<'w, R>;
	type Id = ();
	type Comb<'w, K: CombKind> = Option<CombCase<'w, Self::Item<'w>, Self::Id>>;
	fn comb<'w, K: CombKind>(param: &'w SystemParamItem<Self::Param>)
		-> Self::Comb<'w, K>
	{
		K::wrap((Res::clone(param), ()))
	}
}

impl PredParam for () {
	type Param = ();
	type Item<'w> = ();
	type Id = ();
	type Comb<'w, K: CombKind> = Option<CombCase<'w, Self::Item<'w>, Self::Id>>;
	fn comb<'w, K: CombKind>(param: &'w SystemParamItem<Self::Param>)
		-> Self::Comb<'w, K>
	{
		K::wrap((*param, ()))
	}
}

impl<A: PredParam, B: PredParam> PredParam for (A, B) {
	type Param = (A::Param, B::Param);
	type Item<'w> = (A::Item<'w>, B::Item<'w>);
	type Id = (A::Id, B::Id);
	type Comb<'w, K: CombKind> = PredPairComb<'w, K, A, B>;
	fn comb<'w, K: CombKind>(param: &'w SystemParamItem<Self::Param>)
		-> Self::Comb<'w, K>
	{
		PredPairComb::new(&param.0, &param.1)
	}
}

impl<P: PredParam, const N: usize> PredParam for [P; N]
where
	P::Id: Ord
{
	type Param = P::Param;
	type Item<'w> = [P::Item<'w>; N];
	type Id = [P::Id; N];
	type Comb<'w, K: CombKind> = PredArrayComb<'w, K, P, N>;
	fn comb<'w, K: CombKind>(param: &'w SystemParamItem<Self::Param>)
		-> Self::Comb<'w, K>
	{
		PredArrayComb::new(param)
	}
}

/// A case of prediction.
pub trait PredItem<'w> {
	type Ref<'i>: Copy/* + std::ops::Deref<Target=Self::Inner>*/;
	type Inner: 'w;
	
	/// Needed because `bevy_ecs::world::Ref` can't be cloned/copied.
	fn into_ref(item: Self) -> Self::Ref<'w>;
	
	/// Whether this item is in need of a prediction update.
	fn is_updated(item: &Self) -> bool;
}

impl<'w, T: 'static> PredItem<'w> for Ref<'w, T> {
	type Ref<'i> = &'i Self::Inner;
	type Inner = T;
	fn into_ref(item: Self) -> Self::Ref<'w> {
		Ref::into_inner(item)
	}
	fn is_updated(item: &Self) -> bool {
		DetectChanges::is_changed(item)
	}
}

impl<'w, R: Resource> PredItem<'w> for Res<'w, R> {
	type Ref<'i> = &'i Self::Inner;
	type Inner = R;
	fn into_ref(item: Self) -> Self::Ref<'w> {
		Res::into_inner(item)
	}
	fn is_updated(item: &Self) -> bool {
		DetectChanges::is_changed(item)
	}
}

impl<'w> PredItem<'w> for () {
	type Ref<'i> = ();
	type Inner = ();
	fn into_ref(item: Self) -> Self::Ref<'w> {
		item
	}
	fn is_updated(_item: &Self) -> bool {
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
	fn into_ref((a, b): Self) -> Self::Ref<'w> {
		(A::into_ref(a), B::into_ref(b))
	}
	fn is_updated((a, b): &Self) -> bool {
		A::is_updated(a) || B::is_updated(b)
	}
}

impl<'w, T, const N: usize> PredItem<'w> for [T; N]
where
	T: PredItem<'w>
{
	type Ref<'i> = [T::Ref<'i>; N];
	type Inner = [T::Inner; N];
	fn into_ref(item: Self) -> Self::Ref<'w> {
		item.map(T::into_ref)
	}
	fn is_updated(item: &Self) -> bool {
		item.iter().any(T::is_updated)
	}
}

/// Collects predictions from "when" systems for later compilation.
pub struct PredState<'p, 'w: 'p, 's: 'p, P: PredParam = ()> {
	state: &'p SystemParamItem<'w, 's, P::Param>,
	node: &'p mut Node<PredStateCase<P::Id>>,
}

impl<'p, 'w: 'p, 's: 'p, P: PredParam> PredState<'p, 'w, 's, P> {
	pub(crate) fn new(
		state: &'p SystemParamItem<'w, 's, P::Param>,
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

impl<'p, 'w: 'p, 's: 'p, P: PredParam> IntoIterator for PredState<'p, 'w, 's, P> {
	type Item = <Self::IntoIter as IntoIterator>::Item;
	type IntoIter = PredCombinator<'p, P>;
	fn into_iter(self) -> Self::IntoIter {
		let iter = P::comb::<CombUpdated>(self.state).into_iter();
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
pub struct PredQuery<'world, 'state, D: PredQueryData> {
    world: UnsafeWorldCell<'world>,
    state: &'state D::Id,
}

impl<'w, D: PredQueryData> PredQuery<'w, '_, D> {
	pub fn get_inner(self) -> D::Output<'w> {
		unsafe {
			// SAFETY: Right now this method consumes `self`. If it could be
			// called multiple times, the returned values would overlap.
			<D as PredQueryData>::get_inner(self.world, *self.state)
		}
	}
}

unsafe impl<D: PredQueryData> SystemParam for PredQuery<'_, '_, D> {
	type State = D::Id;
	type Item<'world, 'state> = PredQuery<'world, 'state, D>;
	fn init_state(world: &mut World, system_meta: &mut SystemMeta) -> Self::State {
		// !!! Check for component access overlap. This isn't safe right now.
		if let Some(PredSystemId(id)) = world.get_resource::<PredSystemId>() {
			if let Some(id) = id.downcast_ref::<D::Id>() {
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

/// Combinator for `PredParam` `Query` implementation.
pub struct QueryComb<'w, K, T, F>
where
	T: Component,
	F: ArchetypeFilter + 'static,
{
	inner: &'w Query<'w, 'w, (Ref<'static, T>, Entity), F>,
	kind: PhantomData<K>,
}

impl<K, T, F> Copy for QueryComb<'_, K, T, F>
where
	T: Component,
	F: ArchetypeFilter + 'static,
{}

impl<K, T, F> Clone for QueryComb<'_, K, T, F>
where
	T: Component,
	F: ArchetypeFilter + 'static,
{
	fn clone(&self) -> Self {
		*self
	}
}

impl<'w, K, T, F> IntoIterator for QueryComb<'w, K, T, F>
where
	K: CombKind,
	T: Component,
	F: ArchetypeFilter + 'static,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = QueryCombIter<'w, K, T, F>;
	fn into_iter(self) -> Self::IntoIter {
		QueryCombIter {
			iter: self.inner.iter_inner(),
			kind: PhantomData,
		}
	}
}

/// `Iterator` of `QueryComb`'s `IntoIterator` implementation.
pub struct QueryCombIter<'w, K, T, F>
where
	T: Component,
	F: ArchetypeFilter + 'static,
{
	iter: QueryIter<'w, 'w, (Ref<'static, T>, Entity), F>,
	kind: PhantomData<K>,
}

impl<'w, K, T, F> Iterator for QueryCombIter<'w, K, T, F>
where
	K: CombKind,
	T: Component,
	F: ArchetypeFilter + 'static,
{
	type Item = CombCase<'w, Ref<'w, T>, Entity>;
	fn next(&mut self) -> Option<Self::Item> {
		while let Some(next) = self.iter.next() {
			let wrap = K::wrap(next);
			if wrap.is_some() {
				return wrap
			}
		}
		None
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		match (K::HAS_DIFF, K::HAS_SAME) {
			(false, false) => (0, Some(0)),
			(true, true) => self.iter.size_hint(),
			_ => (0, self.iter.size_hint().1)
		}
	}
}

/// Combinator for `PredParam` tuple implementation.
pub struct PredPairComb<'w, K, A, B>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	a_comb: A::Comb<'w, K>,
	b_comb: B::Comb<'w, K::Pal>,
	a_inv_comb: A::Comb<'w, K::Inv>,
	b_inv_comb: B::Comb<'w, <<K::Inv as CombKind>::Pal as CombKind>::Inv>,
}

impl<K, A, B> Clone for PredPairComb<'_, K, A, B>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	fn clone(&self) -> Self {
		Self {
			a_comb: self.a_comb.clone(),
			b_comb: self.b_comb.clone(),
			a_inv_comb: self.a_inv_comb.clone(),
			b_inv_comb: self.b_inv_comb.clone(),
		}
	}
}

impl<'p, K, A, B> PredPairComb<'p, K, A, B>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	fn new(
		a_param: &'p SystemParamItem<A::Param>,
		b_param: &'p SystemParamItem<B::Param>,
	) -> Self {
		Self {
			a_comb: A::comb(a_param),
			b_comb: B::comb(b_param),
			a_inv_comb: A::comb(a_param),
			b_inv_comb: B::comb(b_param),
		}
	}
}

impl<'w, K, A, B> IntoIterator for PredPairComb<'w, K, A, B>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	type Item = CombCase<'w, <(A, B) as PredParam>::Item<'w>, <(A, B) as PredParam>::Id>;
	type IntoIter = PredPairCombIter<'w, K, A, B>;
	fn into_iter(self) -> Self::IntoIter {
		let Self {
			a_comb,
			b_comb,
			a_inv_comb,
			b_inv_comb,
		} = self;
		PredPairCombIter::primary_next(
			a_comb.into_iter(),
			b_comb,
			a_inv_comb,
			b_inv_comb,
		)
	}
}

/// Iterator for 2-tuple [`PredParam`] types.
pub enum PredPairCombIter<'w, K, A, B>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	Empty,
	Primary {
		a_iter: <A::Comb<'w, K> as IntoIterator>::IntoIter,
		a_curr: <A::Comb<'w, K> as IntoIterator>::Item,
		b_comb: B::Comb<'w, K::Pal>,
		b_iter: <B::Comb<'w, K::Pal> as IntoIterator>::IntoIter,
		a_inv_comb: A::Comb<'w, K::Inv>,
		b_inv_comb: B::Comb<'w, <<K::Inv as CombKind>::Pal as CombKind>::Inv>,
	},
	Secondary {
		b_iter: <B::Comb<'w, <<K::Inv as CombKind>::Pal as CombKind>::Inv> as IntoIterator>::IntoIter,
		b_curr: <B::Comb<'w, <<K::Inv as CombKind>::Pal as CombKind>::Inv> as IntoIterator>::Item,
		a_comb: A::Comb<'w, K::Inv>,
		a_iter: <A::Comb<'w, K::Inv> as IntoIterator>::IntoIter,
	},
}

impl<'w, K, A, B> PredPairCombIter<'w, K, A, B>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	fn primary_next(
		mut a_iter: <A::Comb<'w, K> as IntoIterator>::IntoIter,
		b_comb: B::Comb<'w, K::Pal>,
		a_inv_comb: A::Comb<'w, K::Inv>,
		b_inv_comb: B::Comb<'w, <<K::Inv as CombKind>::Pal as CombKind>::Inv>,
	) -> Self {
		if let Some(a_curr) = a_iter.next() {
			let b_iter = b_comb.clone().into_iter();
			return Self::Primary {
				a_iter,
				a_curr,
				b_comb,
				b_iter,
				a_inv_comb,
				b_inv_comb,
			}
		}
		Self::secondary_next(b_inv_comb.into_iter(), a_inv_comb)
	}
	
	fn secondary_next(
		mut b_iter: <B::Comb<'w, <<K::Inv as CombKind>::Pal as CombKind>::Inv> as IntoIterator>::IntoIter,
		a_comb: A::Comb<'w, K::Inv>,
	) -> Self {
		if let Some(b_curr) = b_iter.next() {
			let a_iter = a_comb.clone().into_iter();
			return Self::Secondary {
				b_iter,
				b_curr,
				a_comb,
				a_iter,
			}
		}
		Self::Empty
	}
}

impl<'w, K, A, B> Iterator for PredPairCombIter<'w, K, A, B>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	type Item = CombCase<'w, <(A, B) as PredParam>::Item<'w>, <(A, B) as PredParam>::Id>;
	fn next(&mut self) -> Option<Self::Item> {
		// !!! Put A/B in order of ascending size to reduce redundancy.
		match std::mem::replace(self, Self::Empty) {
			Self::Empty => None,
			
			 // (Updated A, All B): 
			Self::Primary {
				a_iter,
				a_curr,
				b_comb,
				mut b_iter,
				a_inv_comb,
				b_inv_comb,
			} => {
				if let Some(b_curr) = b_iter.next() {
					*self = Self::Primary {
						a_iter,
						a_curr,
						b_comb,
						b_iter,
						a_inv_comb,
						b_inv_comb,
					};
					let (a, a_id) = a_curr.into_inner();
					let (b, b_id) = b_curr.into_inner();
					return Some(if a_curr.is_diff() || b_curr.is_diff() {
						CombCase::Diff((a, b), (a_id, b_id))
					} else {
						CombCase::Same((a, b), (a_id, b_id))
					})
				}
				*self = Self::primary_next(a_iter, b_comb, a_inv_comb, b_inv_comb);
				self.next()
			}
			
			 // (Updated B, Non-updated A):
			Self::Secondary {
				b_iter,
				b_curr,
				a_comb,
				mut a_iter,
			} => {
				if let Some(a_curr) = a_iter.next() {
					*self = Self::Secondary {
						b_iter,
						b_curr,
						a_comb,
						a_iter,
					};
					let (a, a_id) = a_curr.into_inner();
					let (b, b_id) = b_curr.into_inner();
					return Some(if a_curr.is_diff() || b_curr.is_diff() {
						CombCase::Diff((a, b), (a_id, b_id))
					} else {
						CombCase::Same((a, b), (a_id, b_id))
					})
				}
				*self = Self::secondary_next(b_iter, a_comb);
				self.next()
			},
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		match self {
			Self::Empty => (0, Some(0)),
			Self::Primary { a_iter, b_comb, b_iter, a_inv_comb, b_inv_comb, .. } => {
				let min = b_iter.size_hint().0;
				let max = a_iter.size_hint().1.and_then(|a_max| {
					let b_max = b_comb.clone().into_iter().size_hint().1?;
					let a_inv_max = a_inv_comb.clone().into_iter().size_hint().1?;
					let b_inv_max = b_inv_comb.clone().into_iter().size_hint().1?;
					std::cmp::max(
						// This may be inaccurate if a new `CombKind` is added.
						// It should work for `K = CombAll|None|Updated|Static`.
						min.checked_add(a_max.checked_mul(b_max)?),
						a_inv_max.checked_mul(b_inv_max)
					)
				});
				(min, max)
			},
			Self::Secondary { b_iter, a_comb, a_iter, .. } => {
				let min = a_iter.size_hint().0;
				let max = b_iter.size_hint().1.and_then(|b_max| {
					let a_max = a_comb.clone().into_iter().size_hint().1?;
					min.checked_add(a_max.checked_mul(b_max)?)
				});
				(min, max)
			},
		}
	}
}

/// Combinator for `PredParam` array implementation.
pub struct PredArrayComb<'w, K, P, const N: usize>
where
	K: CombKind,
	P: PredParam,
{
	slice: Box<[<P::Comb<'w, K> as IntoIterator>::Item]>,
}

impl<K: CombKind, P: PredParam, const N: usize> Clone for PredArrayComb<'_, K, P, N> {
	fn clone(&self) -> Self {
		Self {
			slice: self.slice.clone(),
		}
	}
}

impl<'p, K, P, const N: usize> PredArrayComb<'p, K, P, N>
where
	K: CombKind,
	P: PredParam,
	P::Id: Ord,
{
	fn new(param: &'p SystemParamItem<P::Param>) -> Self {
		let mut vec = P::comb::<CombAll>(param).into_iter().collect::<Vec<_>>();
		vec.sort_unstable_by_key(|x| x.id());
		Self {
			slice: vec.into_boxed_slice()
		}
	}
}

impl<'w, K, P, const N: usize> IntoIterator for PredArrayComb<'w, K, P, N>
where
	K: CombKind,
	P: PredParam,
	P::Id: Ord,
{
	type Item = CombCase<'w, <[P; N] as PredParam>::Item<'w>, <[P; N] as PredParam>::Id>;
	type IntoIter = PredArrayCombIter<'w, K, P, N>;
	fn into_iter(self) -> Self::IntoIter {
		let mut iter = PredArrayCombIter {
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
pub struct PredArrayCombIter<'w, K, P, const N: usize>
where
	K: CombKind,
	P: PredParam,
{
	slice: Box<[<P::Comb<'w, CombAll> as IntoIterator>::Item]>,
	index: [usize; N],
	is_first: bool,
	kind: PhantomData<K>,
}

impl<'w, K, P, const N: usize> PredArrayCombIter<'w, K, P, N>
where
	K: CombKind,
	P: PredParam,
	P::Id: Ord,
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

impl<'w, K, P, const N: usize> Iterator for PredArrayCombIter<'w, K, P, N>
where
	K: CombKind,
	P: PredParam,
	P::Id: Ord,
{
	type Item = CombCase<'w, <[P; N] as PredParam>::Item<'w>, <[P; N] as PredParam>::Id>;
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
pub struct PredCombinator<'p, P: PredParam> {
	iter: <P::Comb<'p, CombUpdated> as IntoIterator>::IntoIter,
	node: NodeWriter<'p, PredStateCase<P::Id>>,
}

impl<'p, P: PredParam> Iterator for PredCombinator<'p, P> {
	type Item = (
		&'p mut PredStateCase<P::Id>,
		<P::Item<'p> as PredItem<'p>>::Ref<'p>
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