use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
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
pub(crate) struct PredSystemId {
	pub id: Box<dyn std::any::Any + Send + Sync>,
	pub misc_id: Box<dyn std::any::Any + Send + Sync>,
}

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
	
	fn wrap<T, I>((item, id): (T, I)) -> Option<PredCombCase<T, I>>
	where
		T: PredItem,
		I: PredId,
	{
		if Self::HAS_DIFF || Self::HAS_SAME {
			if T::is_updated(&item) {
				if Self::HAS_DIFF {
					return Some(PredCombCase(T::into_ref(item), id))
				}
			} else if Self::HAS_SAME {
				return Some(PredCombCase(T::into_ref(item), id))
			}
		}
		None
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

/// Linear collections of [`PredParam`] types.
pub trait PredParamVec: PredParam {
	type Head: PredParam;
	type Tail: PredParam;
	
	fn split<'p, 'w, 's>(
		state: &'p SystemParamItem<'w, 's, Self::Param>
	) -> (
		&'p SystemParamItem<'w, 's, <Self::Head as PredParam>::Param>,
		&'p SystemParamItem<'w, 's, <Self::Tail as PredParam>::Param>,
	);
	
	type Split<'p>: Iterator<Item = (
		<<Self::Head as PredParam>::Comb<'p> as PredComb>::Case,
		<Self::Tail as PredParam>::Comb<'p>,
	)>;
	
	fn split2(comb: Self::Comb<'_>) -> Self::Split<'_>;
	
	fn join_id(
		a: <Self::Head as PredParam>::Id,
		b: <Self::Tail as PredParam>::Id,
	) -> Self::Id;
}

// impl<A: PredParam, B: PredParam> PredParamVec for (A, B) {
// 	type Head = A;
// 	type Tail = B;
// 	
// 	fn split<'p, 'w, 's>(
// 		(a, b): &'p SystemParamItem<'w, 's, Self::Param>
// 	) -> (
// 		&'p SystemParamItem<'w, 's, <Self::Head as PredParam>::Param>,
// 		&'p SystemParamItem<'w, 's, <Self::Tail as PredParam>::Param>,
// 	) {
// 		(a, b)
// 	}
// 	
// 	type Split<'p> = Iterator<(
// 		<<Self::Head as PredParam>::Comb<'p> as PredComb>::Case,
// 		<Self::Tail as PredParam>::Comb<'p>,
// 	)>;
// 	
// 	fn split2(comb: Self::Comb<'_>) -> Self::Split<'_> {
// 		todo!()
// 	}
// 	
// 	fn join_id(
// 		a: <Self::Head as PredParam>::Id,
// 		b: <Self::Tail as PredParam>::Id,
// 	) -> Self::Id {
// 		(a, b)
// 	}
// }

// macro_rules! impl_pred_param_vec_for_array {
// 	($size:literal) => {
// 		impl<P: PredParam> PredParamVec for [P; $size]
// 		where
// 			P::Id: Ord,
// 		{
// 			type Head = P;
// 			type Tail = [P; { $size - 1 }];
// 			
// 			fn split<'p, 'w, 's>(
// 				state: &'p SystemParamItem<'w, 's, Self::Param>
// 			) -> (
// 				&'p SystemParamItem<'w, 's, <Self::Head as PredParam>::Param>,
// 				&'p SystemParamItem<'w, 's, <Self::Tail as PredParam>::Param>,
// 			) {
// 				(state, state)
// 			}
// 			
// 			fn join_id(
// 				a: <Self::Head as PredParam>::Id,
// 				b: <Self::Tail as PredParam>::Id,
// 			) -> Self::Id {
// 				let mut array = [a; $size];
// 				array[1..].copy_from_slice(&b);
// 				array
// 			}
// 		}
// 	};
// 	($($size:literal),+) => {
// 		$(impl_pred_param_vec_for_array!($size);)+
// 	};
// }
// 
// impl_pred_param_vec_for_array!(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

/// Iterator of [`PredSubState::iter_step`].
pub struct PredSubStateSplitIter<'p, 'w, 's, P, M, K>
where
	'w: 'p,
	's: 'p,
	P: PredParamVec,
	M: PredId,
	K: CombKind,
{
	state: &'p SystemParamItem<'w, 's, <P::Tail as PredParam>::Param>,
	misc_state: Box<[M]>,
	branches: NodeWriter<'p, PredNodeBranch<'s, P, M>>,
	iter: <<<P::Head as PredParam>::Comb<'p> as PredComb>::IntoKind<K> as IntoIterator>::IntoIter,
	inv_iter: <<<P::Head as PredParam>::Comb<'p> as PredComb>::IntoKind<K::Inv> as IntoIterator>::IntoIter,
}

impl<'p, 'w, 's, P, M, K> Iterator for PredSubStateSplitIter<'p, 'w, 's, P, M, K>
where
	'w: 'p,
	's: 'p,
	P: PredParamVec,
	M: PredId,
	K: CombKind,
{
	type Item = (
		PredSubStateSplit<'p, 's, P::Tail, M, K>,
		<PredParamItem<'p, P::Head> as PredItem>::Ref,
	);
	fn next(&mut self) -> Option<Self::Item> {
		if let Some(case) = self.iter.next() {
			let sub_state = PredSubStateSplit::Main(PredSubState::new(
				P::Tail::comb(self.state).into_kind(),
				self.misc_state.clone(),
				&mut self.branches.write((case.id(), PredNode::Blank)).1,
			));
			return Some((sub_state, case.item_ref()))
		}
		if let Some(case) = self.inv_iter.next() {
			let sub_state = PredSubStateSplit::Pal(PredSubState::new(
				P::Tail::comb(self.state).into_kind(),
				self.misc_state.clone(),
				&mut self.branches.write((case.id(), PredNode::Blank)).1,
			));
			return Some((sub_state, case.item_ref()))
		}
		None
	}
}

/// Nested state of each [`PredSubState::iter_step`].
pub enum PredSubStateSplit<'p, 's, P, M, K>
where
	's: 'p,
	P: PredParam,
	K: CombKind,
{
	Main(PredSubState<'p, 's, P, M, K::Pal>),
	Pal(PredSubState<'p, 's, P, M, <<K::Inv as CombKind>::Pal as CombKind>::Inv>),
}

impl<'p, 's, P, M, K> IntoIterator for PredSubStateSplit<'p, 's, P, M, K>
where
	's: 'p,
	P: PredParam,
	M: PredId,
	K: CombKind,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = PredCombinatorSplit<'p, P, M, K>;
	fn into_iter(self) -> Self::IntoIter {
		match self {
			Self::Main(state) => PredCombinatorSplit::Main(state.into_iter()),
			Self::Pal(state) => PredCombinatorSplit::Pal(state.into_iter()),
		}
	}
}

/// Iterator of [`PredSubStateSplit`].
pub enum PredCombinatorSplit<'p, P: PredParam, M: PredId, K: CombKind> {
	Main(PredCombinator<'p, P, M, K::Pal>),
	Pal(PredCombinator<'p, P, M, <<K::Inv as CombKind>::Pal as CombKind>::Inv>),
}

impl<'p, P, M, K> Iterator for PredCombinatorSplit<'p, P, M, K>
where
	P: PredParam,
	M: PredId,
	K: CombKind,
{
	type Item = (
		&'p mut PredStateCase<P::Id, M>,
		<PredParamItem<'p, P> as PredItem>::Ref
	);
	fn next(&mut self) -> Option<Self::Item> {
		match self {
			Self::Main(iter) => iter.next(),
			Self::Pal(iter) => iter.next(),
		}
	}
}

/// Item of a [`PredComb`]'s iterator.
pub trait PredCombinatorCase: Copy + Clone {
	type Item: PredItem;
	type Id: PredId;
	fn item_ref(&self) -> <Self::Item as PredItem>::Ref;
	fn id(&self) -> Self::Id;
	fn into_parts(self) -> (<Self::Item as PredItem>::Ref, Self::Id) {
		(self.item_ref(), self.id())
	}
}

impl PredCombinatorCase for () {
	type Item = ();
	type Id = ();
	fn item_ref(&self) -> <Self::Item as PredItem>::Ref {}
	fn id(&self) -> Self::Id {}
}

impl<P: PredItem, I: PredId> PredCombinatorCase for PredCombCase<P, I> {
	type Item = P;
	type Id = I;
	fn item_ref(&self) -> <Self::Item as PredItem>::Ref {
		self.0
	}
	fn id(&self) -> Self::Id {
		self.1
	}
}

impl<C: PredCombinatorCase, const N: usize> PredCombinatorCase for [C; N] {
	type Item = [C::Item; N];
	type Id = [C::Id; N];
	fn item_ref(&self) -> <Self::Item as PredItem>::Ref {
		self.map(|x| x.item_ref())
	}
	fn id(&self) -> Self::Id {
		self.map(|x| x.id())
	}
}

impl<A, B> PredCombinatorCase for (A, B)
where
	A: PredCombinatorCase,
	B: PredCombinatorCase,
{
	type Item = (A::Item, B::Item);
	type Id = (A::Id, B::Id);
	fn item_ref(&self) -> <Self::Item as PredItem>::Ref {
		(self.0.item_ref(), self.1.item_ref())
	}
	fn id(&self) -> Self::Id {
		(self.0.id(), self.1.id())
	}
}

/// An item & ID pair of a `PredParam`, with their updated state.
pub struct PredCombCase<P: PredItem, I: PredId>(P::Ref, I);

impl<P: PredItem, I: PredId> Copy for PredCombCase<P, I> {}

impl<P: PredItem, I: PredId> Clone for PredCombCase<P, I> {
	fn clone(&self) -> Self {
		*self
	}
}

/// Combinator type produced by `PredParam::comb`.
pub trait PredComb<K: CombKind = CombAll>: Clone + IntoIterator<Item=Self::Case> {
	type Id: PredId;
	type Case: PredCombinatorCase<Id=Self::Id>;
	
	type IntoKind<Kind: CombKind>:
		PredComb<Kind, Case=Self::Case, Id=Self::Id>;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind>;
}

impl<K: CombKind, T: PredItem> PredComb<K> for Option<PredCombCase<T, ()>> {
	type Id = ();
	type Case = PredCombCase<T, ()>;
	
	type IntoKind<Kind: CombKind> = Self;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind> {
		self
	}
}

impl<'w, T, F, K> PredComb<K> for QueryComb<'w, T, F, K>
where
	K: CombKind,
	T: Component,
	F: ArchetypeFilter + 'static,
{
	type Id = Entity;
	type Case = PredCombCase<Ref<'w, T>, Entity>;
	
	type IntoKind<Kind: CombKind> = QueryComb<'w, T, F, Kind>;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind> {
		QueryComb {
			inner: self.inner,
			kind: PhantomData,
		}
	}
}

impl<A, B, K> PredComb<K> for PredPairComb<A, B, K>
where
	K: CombKind,
	A: PredComb<K>,
	B: PredComb<K::Pal>,
{
	type Id = (A::Id, B::Id);
	type Case = (A::Case, B::Case);
	
	type IntoKind<Kind: CombKind> = PredPairComb<
		A::IntoKind<Kind>,
		B::IntoKind<Kind::Pal>,
		Kind
	>;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind> {
		PredPairComb::new(
			self.a_comb.into_kind(),
			self.b_comb.into_kind(),
		)
	}
}

impl<C, const N: usize, K> PredComb<K> for PredArrayComb<C, N, K>
where
	K: CombKind,
	C: PredComb<K::Pal>,
	C::Id: Ord,
{
	type Id = [C::Id; N];
	type Case = [C::Case; N];
	
	type IntoKind<Kind: CombKind> = PredArrayComb<C::IntoKind<Kind::Pal>, N, Kind>;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind> {
		PredArrayComb {
			slice: self.slice,
		}
	}
}

/// Shortcut for accessing `PredParam::Comb::Case::Item`.
pub type PredParamItem<'w, P> = <<<P
	as PredParam>::Comb<'w>
	as PredComb>::Case
	as PredCombinatorCase>::Item;

/// A set of [`PredItem`] values used to predict & schedule events.
pub trait PredParam {
	/// The equivalent [`bevy_ecs::system::SystemParam`].
	type Param: ReadOnlySystemParam + 'static;
	
	/// Unique identifier for each of [`Self::Param`]'s items.
	type Id: PredId;
	
	/// Creates combinator iterators over [`Self::Param`]'s items.
	type Comb<'w>: PredComb<Id=Self::Id>;
	
	/// Produces [`Self::Comb`].
	fn comb<'w>(param: &'w SystemParamItem<Self::Param>) -> Self::Comb<'w>;
}

impl<T: Component, F: ArchetypeFilter + 'static> PredParam for Query<'_, '_, &T, F> {
	type Param = Query<'static, 'static, (Ref<'static, T>, Entity), F>;
	type Id = Entity;
	type Comb<'w> = QueryComb<'w, T, F>;
	fn comb<'w>(param: &'w SystemParamItem<Self::Param>) -> Self::Comb<'w> {
		QueryComb {
			inner: param,
			kind: PhantomData,
		}
	}
}

impl<R: Resource> PredParam for Res<'_, R> {
	type Param = Res<'static, R>;
	type Id = ();
	type Comb<'w> = Option<PredCombCase<Res<'w, R>, ()>>;
	fn comb<'w>(param: &'w SystemParamItem<Self::Param>) -> Self::Comb<'w> {
		CombAll::wrap((Res::clone(param), ()))
	}
}

impl PredParam for () {
	type Param = ();
	type Id = ();
	type Comb<'w> = Option<PredCombCase<(), ()>>;
	fn comb<'w>(param: &'w SystemParamItem<Self::Param>) -> Self::Comb<'w> {
		CombAll::wrap((*param, ()))
	}
}

impl<A: PredParam, B: PredParam> PredParam for (A, B) {
	type Param = (A::Param, B::Param);
	type Id = (A::Id, B::Id);
	type Comb<'w> = PredPairComb<A::Comb<'w>, B::Comb<'w>>;
	fn comb<'w>((a, b): &'w SystemParamItem<Self::Param>) -> Self::Comb<'w> {
		PredPairComb::new(A::comb(a), B::comb(b))
	}
}

impl<P: PredParam, const N: usize> PredParam for [P; N]
where
	P::Id: Ord,
{
	type Param = P::Param;
	type Id = [P::Id; N];
	type Comb<'w> = PredArrayComb<P::Comb<'w>, N>;
	fn comb<'w>(param: &'w SystemParamItem<Self::Param>) -> Self::Comb<'w> {
		PredArrayComb::new::<P>(param)
	}
}

/// A case of prediction.
pub trait PredItem {
	type Ref: Copy/* + std::ops::Deref<Target=Self::Inner>*/;
	type Inner;
	
	/// Needed because `bevy_ecs::world::Ref` can't be cloned/copied.
	fn into_ref(item: Self) -> Self::Ref;
	
	/// Whether this item is in need of a prediction update.
	fn is_updated(item: &Self) -> bool;
}

impl<'w, T: 'static> PredItem for Ref<'w, T> {
	type Ref = &'w Self::Inner;
	type Inner = T;
	fn into_ref(item: Self) -> Self::Ref {
		Ref::into_inner(item)
	}
	fn is_updated(item: &Self) -> bool {
		DetectChanges::is_changed(item)
	}
}

impl<'w, R: Resource> PredItem for Res<'w, R> {
	type Ref = &'w Self::Inner;
	type Inner = R;
	fn into_ref(item: Self) -> Self::Ref {
		Res::into_inner(item)
	}
	fn is_updated(item: &Self) -> bool {
		DetectChanges::is_changed(item)
	}
}

impl PredItem for () {
	type Ref = ();
	type Inner = ();
	fn into_ref(item: Self) -> Self::Ref {
		item
	}
	fn is_updated(_item: &Self) -> bool {
		true
	}
}

impl<A, B> PredItem for (A, B)
where
	A: PredItem,
	B: PredItem,
{
	type Ref = (A::Ref, B::Ref);
	type Inner = (A::Inner, B::Inner);
	fn into_ref((a, b): Self) -> Self::Ref {
		(A::into_ref(a), B::into_ref(b))
	}
	fn is_updated((a, b): &Self) -> bool {
		A::is_updated(a) || B::is_updated(b)
	}
}

impl<T, const N: usize> PredItem for [T; N]
where
	T: PredItem
{
	type Ref = [T::Ref; N];
	type Inner = [T::Inner; N];
	fn into_ref(item: Self) -> Self::Ref {
		item.map(T::into_ref)
	}
	fn is_updated(item: &Self) -> bool {
		item.iter().any(T::is_updated)
	}
}

/// Collects predictions from "when" systems for later compilation. More general
/// form of [`PredState`] for stepping through combinators layer-wise.
pub struct PredSubState<'p, 's, P, M, K>
where
	's: 'p,
	P: PredParam,
	K: CombKind,
{
	comb: <P::Comb<'p> as PredComb>::IntoKind<K>,
	misc_state: Box<[M]>,
	node: &'p mut PredNode<'s, P, M>,
}

impl<'p, 's, P, M, K> PredSubState<'p, 's, P, M, K>
where
	's: 'p,
	P: PredParam,
	M: PredId,
	K: CombKind,
{
	fn new(
		comb: <P::Comb<'p> as PredComb>::IntoKind<K>,
		misc_state: Box<[M]>, 
		node: &'p mut PredNode<'s, P, M>,
	) -> Self {
		Self {
			comb,
			misc_state,
			node
		}
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

// impl<'p, 'w, 's, P, M, K> PredSubState<'p, 'w, 's, P, M, K>
// where
// 	'w: 'p,
// 	's: 'p,
// 	P: PredParamVec,
// 	M: PredId,
// 	K: CombKind,
// {
// 	pub fn iter_step(self) -> PredSubStateSplitIter<'p, 'w, 's, P, M, K> {
// 		let PredSubState {
// 			state,
// 			misc_state,
// 			node,
// 			..
// 		} = self;
// 		let (head_state, tail_state) = P::split(state);
// 		let iter = P::Head::comb::<K>(head_state).into_iter();
// 		let inv_iter = P::Head::comb::<K::Inv>(head_state).into_iter();
// 		let capacity = 4 * (iter.size_hint().0 + inv_iter.size_hint().0).max(1);
// 		PredSubStateSplitIter {
// 			state: tail_state,
// 			misc_state,
// 			branches: node.init_branches(capacity),
// 			iter,
// 			inv_iter,
// 		}
// 	}
// }

impl<'p, 's, P, M, K> IntoIterator for PredSubState<'p, 's, P, M, K>
where
	's: 'p,
	P: PredParam,
	M: PredId,
	K: CombKind,
{
	type Item = <Self::IntoIter as IntoIterator>::Item;
	type IntoIter = PredCombinator<'p, P, M, K>;
	fn into_iter(self) -> Self::IntoIter {
		let mut iter = self.comb.into_iter();
		let node = self.node.init_data(4 * iter.size_hint().0.max(1));
		let curr = iter.next();
		PredCombinator {
			iter,
			curr,
			misc_state: self.misc_state,
			misc_index: 0,
			node,
		}
	}
}

/// Collects predictions from "when" systems for later compilation.
pub struct PredState<'p, 's, P = (), M = ()>
where
	's: 'p,
	P: PredParam,
	M: PredId,
{
	inner: PredSubState<'p, 's, P, M, CombUpdated>,
}

impl<'p, 's, P, M> PredState<'p, 's, P, M>
where
	's: 'p,
	P: PredParam,
	M: PredId,
{
	pub(crate) fn new(
		comb: <P::Comb<'p> as PredComb>::IntoKind<CombUpdated>,
		misc_state: Box<[M]>, 
		node: &'p mut PredNode<'s, P, M>,
	) -> Self {
		Self {
			inner: PredSubState::new(comb, misc_state, node),
		}
	}
}

// impl<'p, 'w, 's, P, M> PredState<'p, 'w, 's, P, M>
// where
// 	'w: 'p,
// 	's: 'p,
// 	P: PredParamVec,
// 	M: PredId,
// {
// 	pub fn iter_step(self) -> PredSubStateSplitIter<'p, 'w, 's, P, M, CombUpdated> {
// 		self.inner.iter_step()
// 	}
// }

impl<'p, 's, P, M> IntoIterator for PredState<'p, 's, P, M>
where
	's: 'p,
	P: PredParam,
	M: PredId,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = <PredSubState<'p, 's, P, M, CombUpdated> as IntoIterator>::IntoIter;
	fn into_iter(self) -> Self::IntoIter {
		self.inner.into_iter()
	}
}

impl<'p, 's, P, M> Deref for PredState<'p, 's, P, M>
where
	's: 'p,
	P: PredParam,
	M: PredId,
{
	type Target = PredSubState<'p, 's, P, M, CombUpdated>;
	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

impl<'p, 's, P, M> DerefMut for PredState<'p, 's, P, M>
where
	's: 'p,
	P: PredParam,
	M: PredId,
{
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.inner
	}
}

/// A scheduled case of prediction, used in [`crate::PredState`].
pub struct PredStateCase<I, M> {
	id: I,
	misc: M,
	times: Option<Box<dyn Iterator<Item = (Duration, Duration)> + Send + Sync>>,
}

impl<I: PredId, M: PredId> PredStateCase<I, M> {
	fn new(id: I, misc: M) -> Self {
		Self {
			id,
			misc,
			times: None,
		}
	}
	
	pub(crate) fn into_parts(self)
		-> ((I, M), Option<Box<dyn Iterator<Item = (Duration, Duration)> + Send + Sync>>)
	{
		((self.id, self.misc), self.times)
	}
	
	pub fn misc_id(&self) -> M {
		self.misc
	}
	
	pub fn set<T>(&mut self, times: TimeRanges<T>)
	where
		TimeRanges<T>: Iterator<Item = (Duration, Duration)> + Send + Sync + 'static
	{
		self.times = Some(Box::new(times));
	}
}

/// A one-way node that either stores an arbitrary amount of data or branches
/// into sub-nodes.
pub enum PredNode<'s, P: PredParam + 's, M> {
	Blank,
	Data(Node<PredStateCase<P::Id, M>>),
	Branches(Box<dyn PredNodeBranches<'s, P, M> + 's>),
}

impl<'s, P: PredParam + 's, M: PredId> PredNode<'s, P, M> {
	fn init_data(&mut self, cap: usize) -> NodeWriter<PredStateCase<P::Id, M>> {
		if let Self::Blank = self {
			*self = Self::Data(Node::with_capacity(cap));
			if let Self::Data(node) = self {
				NodeWriter::new(node)
			} else {
				unreachable!()
			}
		} else {
			panic!("expected a Blank variant");
		}
	}
	
	fn init_branches(&mut self, cap: usize) -> NodeWriter<PredNodeBranch<'s, P, M>>
	where
		P: PredParamVec
	{
		if let Self::Blank = self {
			*self = Self::Branches(Box::new(Node::with_capacity(cap)));
			if let Self::Branches(branches) = self {
				branches.as_writer()
			} else {
				unreachable!()
			}
		} else {
			panic!("expected a Blank variant");
		}
	}
}

impl<'s, P: PredParam, M: PredId> IntoIterator for PredNode<'s, P, M> {
	type Item = PredStateCase<P::Id, M>;
	type IntoIter = PredNodeIter<P, M>;
	fn into_iter(self) -> Self::IntoIter {
		match self {
			Self::Blank => PredNodeIter::Blank,
			Self::Data(node) => PredNodeIter::Data(node.into_iter()),
			Self::Branches(mut branches) => {
				let mut branch_iter = branches.into_branch_iter();
				if let Some(branch) = branch_iter.next() {
					PredNodeIter::Branches { branch_iter, branch }
				} else {
					PredNodeIter::Blank
				}
			},
		}
	}
}

/// Iterator of [`PredNode`]'s items.
pub enum PredNodeIter<P: PredParam, M> {
	Blank,
	Data(NodeIter<PredStateCase<P::Id, M>>),
	Branches {
		branch_iter: std::vec::IntoIter<std::vec::IntoIter<PredStateCase<P::Id, M>>>,
		branch: std::vec::IntoIter<PredStateCase<P::Id, M>>,
	}
}

impl<P: PredParam, M: PredId> Iterator for PredNodeIter<P, M> {
	type Item = PredStateCase<P::Id, M>;
	fn next(&mut self) -> Option<Self::Item> {
		match self {
			Self::Blank => None,
			Self::Data(iter) => iter.next(),
			Self::Branches { branch_iter, branch } => {
				if let Some(case) = branch.next() {
					Some(case)
				} else if let Some(next_branch) = branch_iter.next() {
					*branch = next_branch;
					self.next()
				} else {
					None
				}
			},
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		match self {
			Self::Blank => (0, Some(0)),
			Self::Data(iter) => iter.size_hint(),
			Self::Branches { branch, .. } => (branch.size_hint().0, None),
		}
	}
}

/// Individual branch of a [`PredNodeBranches`] type.
type PredNodeBranch<'s, P, M> = (
	<<P as PredParamVec>::Head as PredParam>::Id,
	PredNode<'s, <P as PredParamVec>::Tail, M>,
);

/// Used to define a trait object for dynamic branching in [`PredNode`], as not
/// all [`PredParam`] types implement [`PredParamVec`].
pub trait PredNodeBranches<'s, P: PredParam, M: PredId> {
	fn as_writer<'n>(&'n mut self) -> NodeWriter<'n, PredNodeBranch<'s, P, M>>
	where
		P: PredParamVec;
	
	fn into_branch_iter(&mut self)
		-> std::vec::IntoIter<std::vec::IntoIter<PredStateCase<P::Id, M>>>;
}

impl<'s, P, M> PredNodeBranches<'s, P, M> for Node<PredNodeBranch<'s, P, M>>
where
	P: PredParamVec + 's,
	M: PredId,
{
	fn as_writer<'n>(&'n mut self) -> NodeWriter<'n, PredNodeBranch<'s, P, M>>
	where
		P: PredParamVec
	{
		NodeWriter::new(self)
	}
	
	fn into_branch_iter(&mut self)
		-> std::vec::IntoIter<std::vec::IntoIter<PredStateCase<P::Id, M>>>
	{
		let node = std::mem::take(self);
		node.into_iter()
			.map(|(id, node)| {
				node.into_iter()
					.map(move |case| PredStateCase {
						id: P::join_id(id, case.id),
						misc: case.misc,
						times: case.times,
					})
					.collect::<Vec<_>>()
					.into_iter()
			})
			.collect::<Vec<_>>()
			.into_iter()
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
pub struct PredQuery<'world, 'state, D: PredQueryData, M = ()> {
    world: UnsafeWorldCell<'world>,
    state: &'state (D::Id, M),
}

impl<'w, D: PredQueryData, M: PredId> PredQuery<'w, '_, D, M> {
	pub fn get_inner(self) -> D::Output<'w> {
		unsafe {
			// SAFETY: Right now this method consumes `self`. If it could be
			// called multiple times, the returned values would overlap.
			<D as PredQueryData>::get_inner(self.world, self.state.0)
		}
	}
	pub fn misc_id(&self) -> M {
		self.state.1
	}
}

unsafe impl<D: PredQueryData, M: PredId> SystemParam for PredQuery<'_, '_, D, M> {
	type State = (D::Id, M);
	type Item<'world, 'state> = PredQuery<'world, 'state, D, M>;
	fn init_state(world: &mut World, system_meta: &mut SystemMeta) -> Self::State {
		// !!! Check for component access overlap. This isn't safe right now.
		if let Some(PredSystemId { id, misc_id }) = world
			.get_resource::<PredSystemId>()
		{
			let id = if let Some(id) = id.downcast_ref::<D::Id>() {
				*id
			} else {
				panic!(
					"!!! parameter is for wrong ID type. got {:?}",
					std::any::type_name::<D::Id>()
				);
			};
			let misc_id = if let Some(misc_id) = misc_id.downcast_ref::<M>() {
				*misc_id
			} else {
				panic!(
					"!!! misc parameter is for wrong ID type. got {:?}",
					std::any::type_name::<M>()
				);
			};
			(id, misc_id)
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
pub struct QueryComb<'w, T, F, K = CombAll>
where
	T: Component,
	F: ArchetypeFilter + 'static,
{
	inner: &'w Query<'w, 'w, (Ref<'static, T>, Entity), F>,
	kind: PhantomData<K>,
}

impl<T, F, K> Copy for QueryComb<'_, T, F, K>
where
	T: Component,
	F: ArchetypeFilter + 'static,
{}

impl<T, F, K> Clone for QueryComb<'_, T, F, K>
where
	T: Component,
	F: ArchetypeFilter + 'static,
{
	fn clone(&self) -> Self {
		*self
	}
}

impl<'w, T, F, K> IntoIterator for QueryComb<'w, T, F, K>
where
	K: CombKind,
	T: Component,
	F: ArchetypeFilter + 'static,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = QueryCombIter<'w, T, F, K>;
	fn into_iter(self) -> Self::IntoIter {
		QueryCombIter {
			iter: self.inner.iter_inner(),
			kind: PhantomData,
		}
	}
}

/// `Iterator` of `QueryComb`'s `IntoIterator` implementation.
pub struct QueryCombIter<'w, T, F, K>
where
	T: Component,
	F: ArchetypeFilter + 'static,
{
	iter: QueryIter<'w, 'w, (Ref<'static, T>, Entity), F>,
	kind: PhantomData<K>,
}

impl<'w, T, F, K> Iterator for QueryCombIter<'w, T, F, K>
where
	K: CombKind,
	T: Component,
	F: ArchetypeFilter + 'static,
{
	type Item = PredCombCase<Ref<'w, T>, Entity>;
	fn next(&mut self) -> Option<Self::Item> {
		for next in self.iter.by_ref() {
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
pub struct PredPairComb<A, B, K = CombAll>
where
	K: CombKind,
	A: PredComb<K>,
	B: PredComb<K::Pal>,
{
	a_comb: A,
	b_comb: B,
	kind: PhantomData<K>,
}

impl<A, B, K> Clone for PredPairComb<A, B, K>
where
	K: CombKind,
	A: PredComb<K>,
	B: PredComb<K::Pal>,
{
	fn clone(&self) -> Self {
		Self {
			a_comb: self.a_comb.clone(),
			b_comb: self.b_comb.clone(),
			kind: PhantomData,
		}
	}
}

impl<A, B, K> PredPairComb<A, B, K>
where
	K: CombKind,
	A: PredComb<K>,
	B: PredComb<K::Pal>,
{
	fn new(a_comb: A, b_comb: B) -> Self {
		Self {
			a_comb,
			b_comb,
			kind: PhantomData
		}
	}
}

impl<A, B, K> IntoIterator for PredPairComb<A, B, K>
where
	K: CombKind,
	A: PredComb<K>,
	B: PredComb<K::Pal>,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = PredPairCombIter<A, B, K>;
	fn into_iter(self) -> Self::IntoIter {
		let Self { a_comb, b_comb, .. } = self;
		let a_inv_comb = a_comb.clone().into_kind();
		let b_inv_comb = b_comb.clone().into_kind();
		PredPairCombIter::primary_next(
			a_comb.into_iter(),
			b_comb,
			a_inv_comb,
			b_inv_comb,
		)
	}
}

/// Iterator for 2-tuple [`PredParam`] types.
pub enum PredPairCombIter<A, B, K>
where
	K: CombKind,
	A: PredComb<K>,
	B: PredComb<K::Pal>,
{
	Empty,
	Primary {
		a_iter: A::IntoIter,
		a_case: A::Case,
		b_comb: B,
		b_iter: B::IntoIter,
		a_inv_comb: A::IntoKind<K::Inv>,
		b_inv_comb: B::IntoKind<<<K::Inv as CombKind>::Pal as CombKind>::Inv>,
	},
	Secondary {
		b_iter: <B::IntoKind<<<K::Inv as CombKind>::Pal as CombKind>::Inv> as IntoIterator>::IntoIter,
		b_case: <B::IntoKind<<<K::Inv as CombKind>::Pal as CombKind>::Inv> as IntoIterator>::Item,
		a_comb: A::IntoKind<K::Inv>,
		a_iter: <A::IntoKind<K::Inv> as IntoIterator>::IntoIter,
	},
}

impl<A, B, K> PredPairCombIter<A, B, K>
where
	K: CombKind,
	A: PredComb<K>,
	B: PredComb<K::Pal>,
{
	fn primary_next(
		mut a_iter: A::IntoIter,
		b_comb: B,
		a_inv_comb: A::IntoKind<K::Inv>,
		b_inv_comb: B::IntoKind<<<K::Inv as CombKind>::Pal as CombKind>::Inv>,
	) -> Self {
		if let Some(a_case) = a_iter.next() {
			let b_iter = b_comb.clone().into_iter();
			if b_iter.size_hint().1 != Some(0) {
				return Self::Primary {
					a_iter, a_case, b_comb, b_iter, a_inv_comb, b_inv_comb
				}
			}
		}
		Self::secondary_next(b_inv_comb.into_iter(), a_inv_comb)
	}
	
	fn secondary_next(
		mut b_iter: <B::IntoKind<<<K::Inv as CombKind>::Pal as CombKind>::Inv> as IntoIterator>::IntoIter,
		a_comb: A::IntoKind<K::Inv>,
	) -> Self {
		if let Some(b_case) = b_iter.next() {
			let a_iter = a_comb.clone().into_iter();
			if a_iter.size_hint().1 != Some(0) {
				return Self::Secondary { b_iter, b_case, a_comb, a_iter }
			}
		}
		Self::Empty
	}
}

impl<A, B, K> Iterator for PredPairCombIter<A, B, K>
where
	K: CombKind,
	A: PredComb<K>,
	B: PredComb<K::Pal>,
{
	type Item = <PredPairComb<A, B, K> as PredComb<K>>::Case;
	fn next(&mut self) -> Option<Self::Item> {
		// !!! Put A/B in order of ascending size to reduce redundancy.
		match std::mem::replace(self, Self::Empty) {
			Self::Empty => None,
			
			Self::Primary {
				a_iter, a_case, b_comb, mut b_iter, a_inv_comb, b_inv_comb
			} => {
				if let Some(b_case) = b_iter.next() {
					*self = Self::Primary {
						a_iter, a_case, b_comb, b_iter, a_inv_comb, b_inv_comb
					};
					return Some((a_case, b_case))
				}
				*self = Self::primary_next(a_iter, b_comb, a_inv_comb, b_inv_comb);
				self.next()
			}
			
			Self::Secondary { b_iter, b_case, a_comb, mut a_iter } => {
				if let Some(a_case) = a_iter.next() {
					*self = Self::Secondary { b_iter, b_case, a_comb, a_iter };
					return Some((a_case, b_case))
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
pub struct PredArrayComb<C, const N: usize, K = CombAll>
where
	K: CombKind,
	C: PredComb<K::Pal>,
{
	slice: Box<[(C::Case, usize)]>,
}

impl<C, const N: usize, K> Clone for PredArrayComb<C, N, K>
where
	K: CombKind,
	C: PredComb<K::Pal>,
{
	fn clone(&self) -> Self {
		Self {
			slice: self.slice.clone(),
		}
	}
}

impl<C, const N: usize, K> PredArrayComb<C, N, K>
where
	K: CombKind,
	C: PredComb<K::Pal>,
	C::Id: Ord,
{
	fn new<'p, P: PredParam<Comb<'p> = C>>(param: &'p SystemParamItem<P::Param>) -> Self {
		let mut vec = P::comb(param).into_kind::<K::Pal>().into_iter()
			.map(|x| (x, usize::MAX))
			.collect::<Vec<_>>();
		
		vec.sort_unstable_by_key(|(x, _)| x.id());
		
		for item in P::comb(param).into_kind::<K>() {
			if let Ok(target) = vec.binary_search_by(|(x, _)| x.id().cmp(&item.id())) {
				vec[target].1 = target;
				let mut i = target;
				while i != 0 {
					i -= 1;
					if vec[i].1 < target {
						break
					}
					vec[i].1 = target;
				}
			}
		}
		// !!! If `P::comb(param).into_kind::<K>()` returns empty, there's nothing to do.
		
		Self {
			slice: vec.into_boxed_slice()
		}
	}
}

impl<C, const N: usize, K> IntoIterator for PredArrayComb<C, N, K>
where
	K: CombKind,
	C: PredComb<K::Pal>,
	C::Id: Ord,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = PredArrayCombIter<C, N, K>;
	fn into_iter(self) -> Self::IntoIter {
		let mut iter = PredArrayCombIter {
			slice: self.slice,
			index: [0; N],
			layer: 0,
		};
		if N == 0 {
			return iter
		}
		if iter.slice.is_empty() || iter.slice[0].1 >= iter.slice.len() {
			iter.index[0] = iter.slice.len();
		} else if iter.slice[0].1 == 0 {
			iter.layer = N - 1;
		}
		for i in 1..N {
			iter.index[N-i - 1] = iter.index[N-i];
			iter.step(N-i - 1);
		}
		iter
	}
}

/// Iterator for array of [`PredParam`] type.
pub struct PredArrayCombIter<C, const N: usize, K>
where
	K: CombKind,
	C: PredComb<K::Pal>,
{
	slice: Box<[(C::Case, usize)]>,
	index: [usize; N],
	layer: usize,
}

impl<C, const N: usize, K> PredArrayCombIter<C, N, K>
where
	K: CombKind,
	C: PredComb<K::Pal>,
	C::Id: Ord,
{
	fn step_index(&mut self, i: usize) -> bool {
		let index = self.index[i] + 1;
		if index >= self.slice.len() {
			self.index[i] = self.slice.len();
			return true
		}
		self.index[i] = match self.layer.cmp(&i) {
			std::cmp::Ordering::Equal => self.slice[index].1,
			std::cmp::Ordering::Less => {
				if index == self.slice[index].1 {
					self.layer = i;
				}
				index
			},
			_ => index
		};
		if self.index[i] >= self.slice.len() {
			self.index[i] = self.slice.len();
			true
		} else {
			false
		}
	}
	
	fn step(&mut self, i: usize) {
		while self.step_index(i) && self.index[N-1] < self.slice.len() {
			if i + 1 >= self.layer {
				self.layer = 0;
				
				 // Jump to End:
				let next = self.index[i + 1] + 1;
				if next < self.slice.len() && self.slice[next].1 >= self.slice.len() {
					self.index[i + 1] = self.slice.len();
				}
			}
			self.step(i + 1);
			self.index[i] = self.index[i + 1];
		}
	}
}

impl<C, const N: usize, K> Iterator for PredArrayCombIter<C, N, K>
where
	K: CombKind,
	C: PredComb<K::Pal>,
	C::Id: Ord,
{
	type Item = <PredArrayComb<C, N, K> as PredComb<K>>::Case;
	fn next(&mut self) -> Option<Self::Item> {
		if N == 0 || self.index[N-1] >= self.slice.len() {
			return None
		}
		let case = self.index.map(|i| self.slice[i].0);
		self.step(0);
		Some(case)
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		// Currently always produces an exact size.
		
		if N == 0 || self.index[N-1] >= self.slice.len() {
			return (0, Some(0))
		}
		
		fn falling_factorial(n: usize, r: usize) -> usize {
			if n < r {
				return 0
			}
			((1+n-r)..=n).product()
		}
		
		let mut tot = 0;
		let mut div = 1;
		for i in 0..N {
			let mut index = self.index[i];
			if i != 0 {
				index += 1;
			}
			let mut remaining = self.slice.len() - index;
			let mut num = falling_factorial(remaining, i + 1);
			if i >= self.layer {
				while index < self.slice.len() {
					if index == self.slice[index].1 {
						remaining -= 1;
					}
					index += 1;
					if index < self.slice.len() {
						index = self.slice[index].1;
					}
				}
				num -= falling_factorial(remaining, i + 1);
			}
			div *= i + 1;
			tot += num / div;
			// https://www.desmos.com/calculator/l6jawvulhk
		}
		
		(tot, Some(tot))
	}
}

/// Produces all case combinations in need of a new prediction, alongside a
/// [`PredStateCase`] for scheduling.
pub struct PredCombinator<'p, P: PredParam, M, K: CombKind> {
	iter: <<P::Comb<'p> as PredComb>::IntoKind<K> as IntoIterator>::IntoIter,
	curr: Option<<<P::Comb<'p> as PredComb>::IntoKind<K> as IntoIterator>::Item>,
	misc_state: Box<[M]>,
	misc_index: usize,
	node: NodeWriter<'p, PredStateCase<P::Id, M>>,
}

impl<'p, P, M, K> Iterator for PredCombinator<'p, P, M, K>
where
	P: PredParam,
	M: PredId,
	K: CombKind,
{
	type Item = (
		&'p mut PredStateCase<P::Id, M>,
		<PredParamItem<'p, P> as PredItem>::Ref
	);
	fn next(&mut self) -> Option<Self::Item> {
		while let Some(case) = self.curr {
			if let Some(misc) = self.misc_state.get(self.misc_index) {
				self.misc_index += 1;
				let (item, id) = case.into_parts();
				return Some((
					self.node.write(PredStateCase::new(id, *misc)),
					item
				))
			}
			self.curr = self.iter.next();
			self.misc_index = 0;
		}
		None
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		let (min, max) = self.iter.size_hint();
		if self.curr.is_some() {
			let misc_len = self.misc_state.len();
			let misc_num = misc_len - self.misc_index;
			(
				(min * misc_len) + misc_num,
				max.map(|x| (x * misc_len) + misc_num)
			)
		} else {
			(0, Some(0))
		}
	}
}

#[test]
fn array_comb() {
	fn test<const N: usize, const R: usize>(update_list: &[usize]) {
		use crate::*;
		
		#[derive(Component, Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
		struct Test(usize);
		
		let mut app = App::new();
		app.insert_resource::<Time>(Time::default());
		app.add_plugins(ChimePlugin);
		
		for i in 0..N {
			app.world.spawn(Test(i));
		}
		
		fn choose(n: usize, r: usize) -> usize {
			if n < r || r == 0 {
				return 0
			}
			((1+n-r)..=n).product::<usize>() / (1..=r).product::<usize>()
		}
		
		let n_choose_r = choose(N, R);
		let update_vec = update_list.to_vec();
		
		app.add_chime_events((move |
			state: PredState<[Query<&Test>; R]>,
			query: Query<&Test>,
			mut index: system::Local<usize>,
		| {
			let mut iter = state.into_iter();
			match *index {
				0 => { // Full
					assert_eq!(iter.size_hint(), (n_choose_r, Some(n_choose_r)));
					let mut n = 0;
					for ((_, mut a), mut b) in iter
						.zip(query.iter_combinations::<R>())
					{
						// This assumes `iter` and `Query::iter_combinations`
						// will always return in the same order.
						a.sort_unstable();
						b.sort_unstable();
						assert_eq!(a, b);
						n += 1;
					}
					assert_eq!(n, n_choose_r);
				},
				1 => { // Empty
					assert_eq!(iter.size_hint(), (0, Some(0)));
					let next = iter.next();
					if next.is_some() {
						println!("> {:?}", next.as_ref().unwrap().1);
					}
					assert!(next.is_none());
				},
				2 => { // Misc
					let count = n_choose_r - choose(N - update_vec.len(), R);
					assert_eq!(iter.size_hint(), (count, Some(count)));
					let mut n = 0;
					for (_, a) in iter {
						assert!(update_vec.iter()
							.any(|i| a.contains(&&Test(*i))));
						n += 1;
					}
					assert_eq!(n, count);
				},
				_ => unimplemented!(),
			}
			*index += 1;
		}).into_events().on_begin(|| {}));
		
		app.world.run_schedule(ChimeSchedule);
		app.world.run_schedule(ChimeSchedule);
		
		for mut test in app.world.query::<&mut Test>()
			.iter_mut(&mut app.world)
		{
			if update_list.contains(&test.0) {
				test.0 = std::hint::black_box(test.0);
			}
		}
		
		app.world.run_schedule(ChimeSchedule);
	}
	
	 // Normal Cases:
	test::<10, 4>(&[0, 4, 7]);
	test::<200, 2>(&[10, 17, 100, 101, 102, 103, 104, 105, 199]);
	
	 // Weird Cases:
	test::<10, 10>(&[]);
	test::<16, 1>(&[]);
	test::<0, 2>(&[]);
	test::<10, 0>(&[]);
	test::<0, 0>(&[]);
}