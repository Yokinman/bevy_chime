use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
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
	
	type Split<'p, K: CombKind>: Iterator<Item = (
		<<Self::Head as PredParam>::Comb<'p> as IntoIterator>::Item,
		PredSubComb<<Self::Tail as PredParam>::Comb<'p>, K>,
	)>;
	
	fn split<K: CombKind>(
		comb: <Self::Comb<'_> as PredComb>::IntoKind<K>
	) -> Self::Split<'_, K>;
	
	fn join_id(
		a: <Self::Head as PredParam>::Id,
		b: <Self::Tail as PredParam>::Id,
	) -> Self::Id;
}

impl<A: PredParam, B: PredParam> PredParamVec for (A, B) {
	type Head = A;
	type Tail = B;
	
	type Split<'p, K: CombKind> = PredPairCombSplit<
		<<A::Comb<'p> as PredComb>::IntoKind<K::Pal> as IntoIterator>::IntoIter,
		B::Comb<'p>,
		K,
	>;
	
	fn split<K: CombKind>(
		comb: <Self::Comb<'_> as PredComb>::IntoKind<K>
	) -> Self::Split<'_, K> {
		PredPairCombSplit {
			a_iter: comb.a_comb.into_kind().into_iter(),
			b_comb: comb.b_comb,
			kind: PhantomData,
		}
	}
	
	fn join_id(
		a: <Self::Head as PredParam>::Id,
		b: <Self::Tail as PredParam>::Id,
	) -> Self::Id {
		(a, b)
	}
}

/// ...
pub struct PredPairCombSplit<A, B, K> {
	a_iter: A,
	b_comb: B,
	kind: PhantomData<K>,
}

impl<A, B, K> Iterator for PredPairCombSplit<A, B, K>
where
	K: CombKind,
	A: Iterator,
	A::Item: PredCombinatorCase,
	B: PredComb,
{
	type Item = (A::Item, PredSubComb<B, K>);
	fn next(&mut self) -> Option<Self::Item> {
		if let Some(case) = self.a_iter.next() {
			let sub_comb = if case.is_diff() {
				PredSubComb::Diff(self.b_comb.clone().into_kind())
			} else {
				PredSubComb::Same(self.b_comb.clone().into_kind())
			};
			Some((case, sub_comb))
		} else {
			None
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.a_iter.size_hint()
	}
}

macro_rules! impl_pred_param_vec_for_array {
	($size:literal) => {
		impl<P: PredParam> PredParamVec for [P; $size]
		where
			P::Id: Ord,
		{
			type Head = P;
			type Tail = [P; { $size - 1 }];
			
			type Split<'p, K: CombKind>
				= PredArrayCombSplit<P::Comb<'p>, { $size - 1 }, K>;
			
			fn split<K: CombKind>(
				comb: <Self::Comb<'_> as PredComb>::IntoKind<K>
			) -> Self::Split<'_, K> {
				PredArrayCombSplit {
					inner: PredArrayComb {
						comb: comb.comb,
						slice: comb.slice,
						index: comb.index,
						min_diff_index: comb.min_diff_index,
						min_same_index: comb.min_same_index,
						max_diff_index: comb.max_diff_index,
						max_same_index: comb.max_same_index,
						kind: PhantomData,
					}
				}
			}
			
			fn join_id(
				a: <Self::Head as PredParam>::Id,
				b: <Self::Tail as PredParam>::Id,
			) -> Self::Id {
				let mut array = [a; $size];
				array[1..].copy_from_slice(&b);
				array
			}
		}
	};
	($($size:literal),+) => {
		$(impl_pred_param_vec_for_array!($size);)+
	};
}

// `[P; 2]` splits into `A` and `[B; 1]`, which is kind of awkward/unintuitive.
// However, I feel like the consistency has generic utility. An alternative
// method to [`PredSubState::iter_step`] might be worthwhile for convenience.
impl_pred_param_vec_for_array!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

/// ...
pub struct PredArrayCombSplit<C, const N: usize, K>
where
	C: PredComb,
{
	inner: PredArrayComb<C, N, K>,
}

impl<C, const N: usize, K> Iterator for PredArrayCombSplit<C, N, K>
where
	K: CombKind,
	C: PredComb,
	C::Id: Ord,
{
	type Item = (C::Case, PredSubComb<PredArrayComb<C, N>, K>);
	fn next(&mut self) -> Option<Self::Item> {
		if let Some(mut max_index) = self.inner.slice.len().checked_sub(N) {
			max_index = max_index.min(match (K::HAS_DIFF, K::HAS_SAME) {
				(true, true) => self.inner.slice.len(),
				(false, false) => 0,
				(true, false) => self.inner.max_diff_index,
				(false, true) => self.inner.max_same_index,
			});
			if self.inner.index >= max_index {
				return None
			}
		} else {
			return None
		};
		if let Some((case, _)) = self.inner.slice.get(self.inner.index) {
			self.inner.index += 1;
			let sub_comb = if case.is_diff() {
				PredSubComb::Diff(self.inner.clone().into_kind())
			} else {
				PredSubComb::Same(self.inner.clone().into_kind())
			};
			Some((*case, sub_comb))
		} else {
			None
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		if let Some(mut max_index) = self.inner.slice.len().checked_sub(N) {
			max_index = max_index.min(match (K::HAS_DIFF, K::HAS_SAME) {
				(true, true) => self.inner.slice.len(),
				(false, false) => 0,
				(true, false) => self.inner.max_diff_index,
				(false, true) => self.inner.max_same_index,
			});
			let num = max_index - self.inner.index;
			(num, Some(num))
		} else {
			(0, Some(0))
		}
	}
}

/// ...
pub enum PredSubComb<C: PredComb, K: CombKind> {
	Diff(C::IntoKind<K::Pal>),
	Same(C::IntoKind<<<K::Inv as CombKind>::Pal as CombKind>::Inv>),
}

/// Iterator of [`PredSubState::iter_step`].
pub struct PredSubStateSplitIter<'p, 's, P, M, K>
where
	's: 'p,
	P: PredParamVec,
	M: PredId,
	K: CombKind,
{
	iter: <P as PredParamVec>::Split<'p, K>,
	misc_state: Box<[M]>,
	branches: NodeWriter<'p, PredNodeBranch<'s, P, M>>,
}

impl<'p, 's, P, M, K> Iterator for PredSubStateSplitIter<'p, 's, P, M, K>
where
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
		if let Some((head, tail)) = self.iter.next() {
			let node = &mut self.branches.write((head.id(), PredNode::Blank)).1;
			let misc_state = self.misc_state.clone();
			let sub_state = match tail {
				PredSubComb::Diff(comb) => PredSubStateSplit::Diff(PredSubState::new(comb, misc_state, node)),
				PredSubComb::Same(comb) => PredSubStateSplit::Same(PredSubState::new(comb, misc_state, node)),
			};
			Some((sub_state, head.item_ref()))
		} else {
			None
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.iter.size_hint()
	}
}

/// Nested state of each [`PredSubState::iter_step`].
pub enum PredSubStateSplit<'p, 's, P, M, K>
where
	's: 'p,
	P: PredParam,
	K: CombKind,
{
	Diff(PredSubState<'p, 's, P, M, K::Pal>),
	Same(PredSubState<'p, 's, P, M, <<K::Inv as CombKind>::Pal as CombKind>::Inv>),
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
			Self::Diff(state) => PredCombinatorSplit::Diff(state.into_iter()),
			Self::Same(state) => PredCombinatorSplit::Same(state.into_iter()),
		}
	}
}

/// Iterator of [`PredSubStateSplit`].
pub enum PredCombinatorSplit<'p, P: PredParam, M: PredId, K: CombKind> {
	Diff(PredCombinator<'p, P, M, K::Pal>),
	Same(PredCombinator<'p, P, M, <<K::Inv as CombKind>::Pal as CombKind>::Inv>),
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
			Self::Diff(iter) => iter.next(),
			Self::Same(iter) => iter.next(),
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		match self {
			Self::Diff(iter) => iter.size_hint(),
			Self::Same(iter) => iter.size_hint(),
		}
	}
}

/// Item of a [`PredComb`]'s iterator.
pub trait PredCombinatorCase: Copy + Clone {
	type Item: PredItem;
	type Id: PredId;
	fn is_diff(&self) -> bool;
	fn item_ref(&self) -> <Self::Item as PredItem>::Ref;
	fn id(&self) -> Self::Id;
	fn into_parts(self) -> (<Self::Item as PredItem>::Ref, Self::Id) {
		(self.item_ref(), self.id())
	}
}

impl PredCombinatorCase for () {
	type Item = ();
	type Id = ();
	fn is_diff(&self) -> bool {
		true
	}
	fn item_ref(&self) -> <Self::Item as PredItem>::Ref {}
	fn id(&self) -> Self::Id {}
}

impl<P: PredItem, I: PredId> PredCombinatorCase for PredCombCase<P, I> {
	type Item = P;
	type Id = I;
	fn is_diff(&self) -> bool {
		match self {
			PredCombCase::Diff(..) => true,
			PredCombCase::Same(..) => false,
		}
	}
	fn item_ref(&self) -> <Self::Item as PredItem>::Ref {
		let (PredCombCase::Diff(item, _) | PredCombCase::Same(item, _)) = self;
		*item
	}
	fn id(&self) -> Self::Id {
		let (PredCombCase::Diff(_, id) | PredCombCase::Same(_, id)) = self;
		*id
	}
}

impl<C: PredCombinatorCase, const N: usize> PredCombinatorCase for [C; N] {
	type Item = [C::Item; N];
	type Id = [C::Id; N];
	fn is_diff(&self) -> bool {
		self.iter().any(C::is_diff)
	}
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
	fn is_diff(&self) -> bool {
		self.0.is_diff() || self.1.is_diff()
	}
	fn item_ref(&self) -> <Self::Item as PredItem>::Ref {
		(self.0.item_ref(), self.1.item_ref())
	}
	fn id(&self) -> Self::Id {
		(self.0.id(), self.1.id())
	}
}

/// An item & ID pair of a `PredParam`, with their updated state.
pub enum PredCombCase<P: PredItem, I: PredId> {
	Diff(P::Ref, I),
	Same(P::Ref, I),
}

impl<P: PredItem, I: PredId> Copy for PredCombCase<P, I> {}

impl<P: PredItem, I: PredId> Clone for PredCombCase<P, I> {
	fn clone(&self) -> Self {
		*self
	}
}

/// Combinator type produced by `PredParam::comb`.
pub trait PredComb<K: CombKind = CombNone>: Clone + IntoIterator<Item=Self::Case> {
	type Id: PredId;
	type Case: PredCombinatorCase<Id=Self::Id>;
	
	type IntoKind<Kind: CombKind>:
		PredComb<Kind, Case=Self::Case, Id=Self::Id>;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind>;
}

impl<K: CombKind> PredComb<K> for EmptyComb<K> {
	type Id = ();
	type Case = PredCombCase<(), ()>;
	
	type IntoKind<Kind: CombKind> = EmptyComb<Kind>;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind> {
		EmptyComb {
			kind: PhantomData,
		}
	}
}

impl<'w, R, K> PredComb<K> for ResComb<'w, R, K>
where
	K: CombKind,
	R: Resource,
{
	type Id = ();
	type Case = PredCombCase<Res<'w, R>, ()>;
	
	type IntoKind<Kind: CombKind> = ResComb<'w, R, Kind>;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind> {
		ResComb {
			inner: Res::clone(&self.inner),
			kind: PhantomData,
		}
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
	A: PredComb,
	B: PredComb,
{
	type Id = (A::Id, B::Id);
	type Case = (A::Case, B::Case);
	
	type IntoKind<Kind: CombKind> = PredPairComb<A, B, Kind>;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind> {
		PredPairComb::new(self.a_comb, self.b_comb)
	}
}

impl<C, const N: usize, K> PredComb<K> for PredArrayComb<C, N, K>
where
	K: CombKind,
	C: PredComb,
	C::Id: Ord,
{
	type Id = [C::Id; N];
	type Case = [C::Case; N];
	
	type IntoKind<Kind: CombKind> = PredArrayComb<C, N, Kind>;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind> {
		if K::Pal::HAS_DIFF == Kind::Pal::HAS_DIFF
			&& K::Pal::HAS_SAME == Kind::Pal::HAS_SAME
		{
			PredArrayComb {
				comb: self.comb,
				slice: self.slice,
				index: self.index,
				min_diff_index: self.min_diff_index,
				min_same_index: self.min_same_index,
				max_diff_index: self.max_diff_index,
				max_same_index: self.max_same_index,
				kind: PhantomData,
			}
		} else {
			PredArrayComb::new(self.comb)
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
	type Comb<'w> = ResComb<'w, R>;
	fn comb<'w>(param: &'w SystemParamItem<Self::Param>) -> Self::Comb<'w> {
		ResComb {
			inner: Res::clone(param),
			kind: PhantomData,
		}
	}
}

impl PredParam for () {
	type Param = ();
	type Id = ();
	type Comb<'w> = EmptyComb;
	fn comb<'w>(_param: &'w SystemParamItem<Self::Param>) -> Self::Comb<'w> {
		EmptyComb {
			kind: PhantomData,
		}
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
		PredArrayComb::new(P::comb(param))
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

impl<'p, 's, P, M, K> PredSubState<'p, 's, P, M, K>
where
	's: 'p,
	P: PredParamVec,
	M: PredId,
	K: CombKind,
{
	pub fn iter_step(self) -> PredSubStateSplitIter<'p, 's, P, M, K> {
		let PredSubState { comb, misc_state, node } = self;
		let iter = P::split(comb);
		let capacity = 4 * iter.size_hint().0.max(1);
		PredSubStateSplitIter {
			iter,
			misc_state,
			branches: node.init_branches(capacity),
		}
	}
}

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

impl<'p, 's, P, M> PredState<'p, 's, P, M>
where
	's: 'p,
	P: PredParamVec,
	M: PredId,
{
	pub fn iter_step(self) -> PredSubStateSplitIter<'p, 's, P, M, CombUpdated> {
		self.inner.iter_step()
	}
}

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
	type IntoIter = PredNodeIter<'s, P, M>;
	fn into_iter(self) -> Self::IntoIter {
		match self {
			Self::Blank => PredNodeIter::Blank,
			Self::Data(node) => PredNodeIter::Data(node.into_iter()),
			Self::Branches(mut branches) => PredNodeIter::Branches(branches.into_branch_iter()),
		}
	}
}

/// Iterator of [`PredNode`]'s items.
pub enum PredNodeIter<'s, P: PredParam, M> {
	Blank,
	Data(NodeIter<PredStateCase<P::Id, M>>),
	Branches(Box<dyn PredNodeBranchesIterator<'s, P, M> + 's>),
}

impl<P: PredParam, M: PredId> Iterator for PredNodeIter<'_, P, M> {
	type Item = PredStateCase<P::Id, M>;
	fn next(&mut self) -> Option<Self::Item> {
		match self {
			Self::Blank => None,
			Self::Data(iter) => iter.next(),
			Self::Branches(iter) => iter.next(),
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		match self {
			Self::Blank => (0, Some(0)),
			Self::Data(iter) => iter.size_hint(),
			Self::Branches(iter) => iter.size_hint(),
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
/// 
/// ??? To avoid dynamic dispatch: could move all `PredParamVec` items into
/// `PredParam` and implement empty defaults for scalar types. However, this
/// would only support a subset of arrays instead of all sizes, which feels
/// like an unnecessary constraint. Specialization would probably help here.
pub trait PredNodeBranches<'s, P: PredParam, M: PredId> {
	fn as_writer<'n>(&'n mut self) -> NodeWriter<'n, PredNodeBranch<'s, P, M>>
	where
		P: PredParamVec;
	
	fn into_branch_iter(&mut self) -> Box<dyn PredNodeBranchesIterator<'s, P, M> + 's>;
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
	
	fn into_branch_iter(&mut self) -> Box<dyn PredNodeBranchesIterator<'s, P, M> + 's> {
		Box::new(PredNodeBranchesIter {
			node_iter: std::mem::take(self).into_iter(),
			branch_id: None,
			branch_iter: PredNodeIter::Blank,
		})
	}
}

/// Specific type of [`PredNodeBranchesIterator`] trait objects.
pub struct PredNodeBranchesIter<'s, P: PredParamVec, M> {
	node_iter: NodeIter<PredNodeBranch<'s, P, M>>,
	branch_id: Option<<P::Head as PredParam>::Id>,
	branch_iter: PredNodeIter<'s, P::Tail, M>,
}

impl<'s, P, M> Iterator for PredNodeBranchesIter<'s, P, M>
where
	P: PredParamVec,
	M: PredId,
{
	type Item = PredStateCase<P::Id, M>;
	fn next(&mut self) -> Option<Self::Item> {
		if let Some(case) = self.branch_iter.next() {
			Some(PredStateCase {
				id: P::join_id(self.branch_id.unwrap(), case.id),
				misc: case.misc,
				times: case.times,
			})
		} else if let Some((id, node)) = self.node_iter.next() {
			self.branch_id = Some(id);
			self.branch_iter = node.into_iter();
			self.next()
		} else {
			None
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		(self.branch_iter.size_hint().0, None)
	}
}

/// Used to define a trait object for dynamic branching in [`PredNodeIter`], as
/// not all [`PredParam`] types implement [`PredParamVec`].
pub trait PredNodeBranchesIterator<'s, P: PredParam, M>:
	Iterator<Item = PredStateCase<P::Id, M>>
{}

impl<'s, P, M> PredNodeBranchesIterator<'s, P, M> for PredNodeBranchesIter<'s, P, M>
where
	P: PredParamVec,
	M: PredId,
{}

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

/// Combinator for `PredParam` `()` implementation.
pub struct EmptyComb<K = CombNone> {
	kind: PhantomData<K>,
}

impl<K> Clone for EmptyComb<K> {
	fn clone(&self) -> Self {
		Self {
			kind: PhantomData,
		}
	}
}

impl<K> IntoIterator for EmptyComb<K>
where
	K: CombKind
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = CombIter<std::iter::Once<((), ())>, K>;
	fn into_iter(self) -> Self::IntoIter {
		CombIter::new(std::iter::once(((), ())))
	}
}

/// Combinator for `PredParam` `Res` implementation.
pub struct ResComb<'w, T, K = CombNone>
where
	T: Resource,
{
	inner: Res<'w, T>,
	kind: PhantomData<K>,
}

impl<T, K> Clone for ResComb<'_, T, K>
where
	T: Resource,
{
	fn clone(&self) -> Self {
		Self {
			inner: Res::clone(&self.inner),
			kind: PhantomData,
		}
	}
}

impl<'w, T, K> IntoIterator for ResComb<'w, T, K>
where
	K: CombKind,
	T: Resource,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = CombIter<std::iter::Once<(Res<'w, T>, ())>, K>;
	fn into_iter(self) -> Self::IntoIter {
		CombIter::new(std::iter::once((self.inner, ())))
	}
}

/// Combinator for `PredParam` `Query` implementation.
pub struct QueryComb<'w, T, F, K = CombNone>
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
	type IntoIter = CombIter<QueryIter<'w, 'w, (Ref<'static, T>, Entity), F>, K>;
	fn into_iter(self) -> Self::IntoIter {
		CombIter::new(self.inner.iter_inner())
	}
}

/// `Iterator` of `ResComb`'s `IntoIterator` implementation.
pub struct CombIter<T, K> {
	iter: T,
	kind: PhantomData<K>,
}

impl<T, K> CombIter<T, K> {
	fn new(iter: T) -> Self {
		Self {
			iter,
			kind: PhantomData,
		}
	}
}

impl<P, I, T, K> Iterator for CombIter<T, K>
where
	P: PredItem,
	I: PredId,
	T: Iterator<Item = (P, I)>,
	K: CombKind,
{
	type Item = PredCombCase<P, I>;
	fn next(&mut self) -> Option<Self::Item> {
		if !K::HAS_DIFF && !K::HAS_SAME {
			return None
		}
		for (item, id) in self.iter.by_ref() {
			if P::is_updated(&item) {
				if K::HAS_DIFF {
					return Some(PredCombCase::Diff(P::into_ref(item), id))
				}
			} else if K::HAS_SAME {
				return Some(PredCombCase::Same(P::into_ref(item), id))
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
pub struct PredPairComb<A, B, K = CombNone> {
	a_comb: A,
	b_comb: B,
	kind: PhantomData<K>,
}

impl<A, B, K> Clone for PredPairComb<A, B, K>
where
	A: Clone,
	B: Clone,
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
	A: PredComb,
	B: PredComb,
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
	A: PredComb,
	B: PredComb,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = PredPairCombIter<A, B, K>;
	fn into_iter(self) -> Self::IntoIter {
		let Self { a_comb, b_comb, .. } = self;
		let a_inv_comb = a_comb.clone().into_kind();
		let b_inv_comb = b_comb.clone().into_kind();
		PredPairCombIter::primary_next(
			a_comb.into_kind().into_iter(),
			b_comb.into_kind(),
			a_inv_comb,
			b_inv_comb,
		)
	}
}

/// Iterator for 2-tuple [`PredParam`] types.
pub enum PredPairCombIter<A, B, K>
where
	K: CombKind,
	A: PredComb,
	B: PredComb,
{
	Empty,
	Primary {
		a_iter: <A::IntoKind<K> as IntoIterator>::IntoIter,
		a_case: A::Case,
		b_comb: B::IntoKind<K::Pal>,
		b_iter: <B::IntoKind<K::Pal> as IntoIterator>::IntoIter,
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
	A: PredComb,
	B: PredComb,
{
	fn primary_next(
		mut a_iter: <A::IntoKind<K> as IntoIterator>::IntoIter,
		b_comb: B::IntoKind<K::Pal>,
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
	A: PredComb,
	B: PredComb,
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
pub struct PredArrayComb<C, const N: usize, K = CombNone>
where
	C: PredComb,
{
	comb: C,
	slice: Rc<[(C::Case, usize)]>,
	index: usize,
	min_diff_index: usize,
	min_same_index: usize,
	max_diff_index: usize,
	max_same_index: usize,
	kind: PhantomData<K>,
}

impl<C, const N: usize, K> Clone for PredArrayComb<C, N, K>
where
	C: PredComb,
{
	fn clone(&self) -> Self {
		Self {
			comb: self.comb.clone(),
			slice: Rc::clone(&self.slice),
			index: self.index,
			min_diff_index: self.min_diff_index,
			min_same_index: self.min_same_index,
			max_diff_index: self.max_diff_index,
			max_same_index: self.max_same_index,
			kind: PhantomData,
		}
	}
}

impl<C, const N: usize, K> PredArrayComb<C, N, K>
where
	K: CombKind,
	C: PredComb,
	C::Id: Ord,
{
	fn new(comb: C) -> Self {
		let mut vec = comb.clone().into_kind::<K::Pal>().into_iter()
			.map(|x| (x, usize::MAX))
			.collect::<Vec<_>>();
		
		vec.sort_unstable_by_key(|(x, _)| x.id());
		
		 // Setup Jump Indices:
		let mut index = vec.len();
		let mut min_diff_index = index;
		let mut min_same_index = index;
		let mut max_diff_index = 0;
		let mut max_same_index = 0;
		while index != 0 {
			index -= 1;
			let (case, next_alt_index) = &mut vec[index];
			if case.is_diff() {
				*next_alt_index = min_diff_index;
				min_diff_index = index;
				if max_diff_index == 0 {
					max_diff_index = index + 1;
				}
			} else {
				*next_alt_index = min_same_index;
				min_same_index = index;
				if max_same_index == 0 {
					max_same_index = index + 1;
				}
			}
		}
		
		Self {
			comb,
			slice: vec.into(),
			index: 0,
			min_diff_index,
			min_same_index,
			max_diff_index,
			max_same_index,
			kind: PhantomData,
		}
	}
}

impl<C, const N: usize, K> IntoIterator for PredArrayComb<C, N, K>
where
	K: CombKind,
	C: PredComb,
	C::Id: Ord,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = PredArrayCombIter<C, N, K>;
	fn into_iter(self) -> Self::IntoIter {
		let mut iter = PredArrayCombIter {
			slice: self.slice,
			index: [self.index; N],
			min_diff_index: self.min_diff_index,
			min_same_index: self.min_same_index,
			layer: 0,
			kind: PhantomData,
		};
		if N == 0 {
			return iter
		}
		
		 // Initialize Main Index:
		if match (K::HAS_DIFF, K::HAS_SAME) {
			(true, true) => false,
			(false, false) => true,
			(true, false) => iter.min_diff_index >= iter.slice.len(),
			(false, true) => iter.min_same_index >= iter.slice.len(),
		} {
			iter.index[N-1] = iter.slice.len();
		} else if iter.index[N-1] < iter.slice.len() {
			let (case, _) = iter.slice[iter.index[N-1]];
			if if case.is_diff() { K::HAS_DIFF } else { K::HAS_SAME } {
				iter.layer = N-1;
			} else if N == 1 {
				iter.step_index(N-1);
			}
		}
		
		 // Initialize Sub-indices:
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
	C: PredComb,
{
	slice: Rc<[(C::Case, usize)]>,
	index: [usize; N],
	min_diff_index: usize,
	min_same_index: usize,
	layer: usize,
	kind: PhantomData<K>,
}

impl<C, const N: usize, K> PredArrayCombIter<C, N, K>
where
	C: PredComb,
	C::Id: Ord,
	K: CombKind,
{
	fn step_index(&mut self, i: usize) -> bool {
		let index = self.index[i];
		if index >= self.slice.len() {
			return true
		}
		self.index[i] = match self.layer.cmp(&i) {
			std::cmp::Ordering::Equal => match (K::HAS_DIFF, K::HAS_SAME){
				(true, true) => index + 1,
				(false, false) => self.slice.len(),
				_ => {
					let (case, next_index) = self.slice[index];
					
					 // Jump to Next Matching Case:
					if if case.is_diff() { K::HAS_DIFF } else { K::HAS_SAME } {
						next_index
					}
					
					 // Find Next Matching Case:
					else {
						let first_index = if K::HAS_DIFF {
							self.min_diff_index
						} else {
							self.min_same_index
						};
						if index < first_index {
							first_index
						} else {
							let mut index = index + 1;
							while let Some((case, _)) = self.slice.get(index) {
								if if case.is_diff() { K::HAS_DIFF } else { K::HAS_SAME } {
									break
								}
								index += 1;
							}
							index
						}
					}
				}
			},
			std::cmp::Ordering::Less => {
				if let Some((case, _)) = self.slice.get(index + 1) {
					if if case.is_diff() { K::HAS_DIFF } else { K::HAS_SAME } {
						self.layer = i;
					}
				}
				index + 1
			},
			_ => index + 1
		};
		self.index[i] >= self.slice.len()
	}
	
	fn step(&mut self, i: usize) {
		while self.step_index(i) && self.index[N-1] < self.slice.len() {
			if i + 1 >= self.layer {
				self.layer = 0;
				
				 // Jump to End:
				if !K::HAS_DIFF || !K::HAS_SAME {
					if self.slice[self.index[i + 1]].1 >= self.slice.len() {
						self.index[i + 1] = self.slice.len();
					}
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
	C: PredComb,
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
		
		if N == 0
			|| self.index[N-1] >= self.slice.len()
			|| (!K::HAS_DIFF && !K::HAS_SAME)
		{
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
			if i >= self.layer && (!K::HAS_DIFF || !K::HAS_SAME) {
				let first_index = if K::HAS_DIFF {
					self.min_diff_index
				} else {
					self.min_same_index
				};
				while index < self.slice.len() {
					let (case, next_index) = self.slice[index];
					index = if if case.is_diff() { K::HAS_DIFF } else { K::HAS_SAME } {
						remaining -= 1;
						next_index
					} else if index < first_index {
						first_index
					} else {
						index + 1
					};
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

#[cfg(test)]
mod testing {
	use super::*;
	
	#[derive(Component, Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
	struct Test(usize);
	
	#[derive(Component, Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
	struct TestB(usize);
	
	fn test_pair<const A: usize, const B: usize>(
		update_list: &[usize],
		b_update_list: &[usize],
	)
	where
		for<'w, 's, 'a, 'w1, 's1, 'a1, 'b>
			(Query<'w, 's, &'a Test>, Query<'w1, 's1, &'a1 TestB>):
				PredParamVec<
					Head = Query<'w, 's, &'a Test>,
					Tail = Query<'w1, 's1, &'a1 TestB>,
					Comb<'b> = PredPairComb<QueryComb<'b, Test, ()>, QueryComb<'b, TestB, ()>>
				>,
	{
		use crate::*;
		
		let mut app = App::new();
		app.insert_resource::<Time>(Time::default());
		app.add_plugins(ChimePlugin);
		
		for i in 0..A {
			app.world.spawn(Test(i));
		}
		for i in 0..B {
			app.world.spawn(TestB(i));
		}
		
		 // Setup [`PredPairComb`] Testing:
		let update_vec = update_list.to_vec();
		let b_update_vec = b_update_list.to_vec();
		app.add_chime_events((move |
			state: PredState<(Query<&Test>, Query<&TestB>)>,
			a_query: Query<&Test>,
			b_query: Query<&TestB>,
			mut index: system::Local<usize>,
		| {
			let mut iter = state.into_iter();
			match *index {
				0 => { // Full
					if A > 1 && B > 1 {
						// !!! This shouldn't really be +1, but it's an issue
						// with how `PredCombinator` adds 1 to the upper bound
						// of `PredPairCombIter` which remains semi-constant:
						assert_eq!(iter.size_hint(), (B, Some(A*B + 1)));
					}
					let mut n = 0;
					for a in &a_query {
						for b in &b_query {
							// This assumes `iter` and `QueryIter` will always
							// produce the same order.
							if let Some((_, next)) = iter.next() {
								assert_eq!(next, (a, b));
							} else {
								panic!();
							}
							n += 1;
						}
					}
					assert_eq!(n, A*B);
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
					let count = A*b_update_vec.len()
						+ B*update_vec.len()
						- update_vec.len()*b_update_vec.len();
					if A > 1 && B > 1 {
						assert_eq!(iter.size_hint(), (B, Some(A*B + 1)));
					}
					let mut n = 0;
					for (_, (a, b)) in iter {
						assert!(update_vec.contains(&a.0)
							|| b_update_vec.contains(&b.0));
						n += 1;
					}
					assert_eq!(n, count);
				},
				_ => unimplemented!(),
			}
			*index += 1;
		}).into_events().on_begin(|| {}));
		
		 // Setup [`PredSubState::iter_step`] Testing:
		let update_vec = update_list.to_vec();
		let b_update_vec = b_update_list.to_vec();
		app.add_chime_events((move |
			state: PredState<(Query<&Test>, Query<&TestB>)>,
			a_query: Query<Ref<Test>>,
			b_query: Query<Ref<TestB>>,
			mut index: system::Local<usize>,
		| {
			let mut iter = state.iter_step();
			match *index {
				0 => { // Full
					// !!! This should be `(0, Some(A))`, and PredPairCombSplit
					// shouldn't return A-values that have an empty B iterator.
					assert_eq!(iter.size_hint(), (A, Some(A)));
					let mut n = 0;
					for a in &a_query {
						if let Some((state, x)) = iter.next() {
							let mut iter = state.into_iter();
							assert_eq!(iter.size_hint(), (B, Some(B)));
							assert_eq!(*x, *a);
							for b in &b_query {
								// This assumes `iter` and `QueryIter` always
								// produce the same order.
								if let Some((_, y)) = iter.next() {
									assert_eq!(*y, *b);
								} else {
									panic!();
								}
								n += 1;
							}
						} else {
							panic!();
						}
					}
					assert_eq!(n, A*B);
				},
				1 => { // Empty
					assert_eq!(iter.size_hint(), (A, Some(A)));
					for (state, _) in iter {
						let mut iter = state.into_iter();
						let next = iter.next();
						if next.is_some() {
							println!("> {:?}", next.as_ref().unwrap().1);
						}
						assert!(next.is_none());
					}
				},
				2 => { // Misc
					let count = A*b_update_vec.len()
						+ B*update_vec.len()
						- update_vec.len()*b_update_vec.len();
					assert_eq!(iter.size_hint(), (A, Some(A)));
					let mut n = 0;
					for ((state, a), a_ref) in iter.zip(&a_query) {
						// This assumes `iter` and `QueryIter` always produce
						// the same order.
						assert_eq!(*a, *a_ref);
						let iter = state.into_iter();
						if DetectChanges::is_changed(&a_ref) {
							assert!(update_vec.contains(&a.0));
							assert_eq!(iter.size_hint(), (B, Some(B)));
							for ((_, b), b_ref) in iter.zip(&b_query) {
								assert_eq!(*b, *b_ref);
								n += 1;
							}
						} else {
							assert!(!update_vec.contains(&a.0));
							for (_, b) in iter {
								assert!(b_update_vec.contains(&b.0));
								n += 1;
							}
						}
					}
					assert_eq!(n, count);
				},
				_ => unimplemented!(),
			}
			*index += 1;
		}).into_events().on_begin(|| {}));
		
		 // Run Tests:
		app.world.run_schedule(ChimeSchedule);
		app.world.run_schedule(ChimeSchedule);
		for mut test in app.world.query::<&mut Test>()
			.iter_mut(&mut app.world)
		{
			if update_list.contains(&test.0) {
				test.0 = std::hint::black_box(test.0);
			}
		}
		for mut test in app.world.query::<&mut TestB>()
			.iter_mut(&mut app.world)
		{
			if b_update_list.contains(&test.0) {
				test.0 = std::hint::black_box(test.0);
			}
		}
		app.world.run_schedule(ChimeSchedule);
	}
	
	fn test_array<const N: usize, const R: usize>(update_list: &[usize])
	where
		for<'w, 's, 'a, 'b> [Query<'w, 's, &'a Test>; R]:
			PredParamVec<
				Head = Query<'w, 's, &'a Test>,
				Comb<'b> = PredArrayComb<QueryComb<'b, Test, ()>, R>
			>,
		for<'w, 's, 'a, 'b>
			<<<<<[Query<'w, 's, &'a Test>; R]
				as PredParamVec>::Tail
				as PredParam>::Comb<'b>
				as PredComb>::Case
				as PredCombinatorCase>::Item
				as PredItem>::Ref:
					IntoIterator,
		for<'w, 's, 'a, 'b>
			<<<<<<[Query<'w, 's, &'a Test>; R]
				as PredParamVec>::Tail
				as PredParam>::Comb<'b>
				as PredComb>::Case
				as PredCombinatorCase>::Item
				as PredItem>::Ref
				as IntoIterator>::Item:
					Deref<Target = Test>,
	{
		use crate::*;
		
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
		
		 // Setup [`PredArrayComb`] Testing:
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
		
		 // Setup [`PredSubState::iter_step`] Testing:
		let update_vec = update_list.to_vec();
		app.add_chime_events((move |
			state: PredState<[Query<&Test>; R]>,
			query: Query<Ref<Test>>,
			mut index: system::Local<usize>,
		| {
			let mut iter = state.iter_step();
			match *index {
				0 => { // Full
					let count = N.checked_sub(R).map(|x| x + 1).unwrap_or(0);
					assert_eq!(iter.size_hint(), (count, Some(count)));
					let mut n = 0;
					for ((state, a), b) in iter.zip(&query) {
						// This assumes `iter` and `Query` will always return
						// in the same order.
						assert_eq!(*a, *b);
						let count = choose(N - (n + 1), R - 1);
						assert_eq!(state.into_iter().size_hint(), (count, Some(count)));
						n += 1;
					}
					assert_eq!(n, count);
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
					let count = update_vec.iter().max().copied()
						.min(N.checked_sub(R))
						.map(|x| x + 1)
						.unwrap_or(0);
					assert_eq!(iter.size_hint(), (count, Some(count)));
					let mut n = 0;
					for ((state, a), b) in iter.zip(&query) {
						// This assumes `iter` and `Query` will always return
						// in the same order.
						assert_eq!(*a, *b);
						if DetectChanges::is_changed(&b) {
							assert!(update_vec.contains(&n));
							let count = choose(N - (n + 1), R - 1);
							assert_eq!(state.into_iter().size_hint(), (count, Some(count)));
						} else {
							for (_, x) in state {
								let list = x.into_iter()
									.map(|x| *x)
									.collect::<Vec<_>>();
								assert!(
									update_vec.iter()
										.any(|i| list.contains(&&Test(*i))),
									"{:?} not in {:?}",
									(list, *a), update_vec,
								);
							}
						}
						n += 1;
					}
					assert_eq!(n, count);
				},
				_ => unimplemented!(),
			}
			*index += 1;
		}).into_events().on_begin(|| {}));
		
		 // Run Tests:
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
	
	#[test]
	fn array_comb() {
		 // Normal Cases:
		test_array::<10, 4>(&[0, 4, 6]);
		test_array::<200, 2>(&[10, 17, 100, 101, 102, 103, 104, 105, 199]);
		
		 // Weird Cases:
		test_array::<10, 10>(&[]);
		test_array::<16, 1>(&[]);
		test_array::<0, 2>(&[]);
		// test_array::<10, 0>(&[]);
		// test_array::<0, 0>(&[]);
	}
	
	#[test]
	fn pair_comb() {
		 // Normal Cases:
		test_pair::<40, 100>(&[0, 1, 5, 20], &[5, 51, 52, 53, 55, 99]);
		test_pair::<100, 40>(&[5, 51, 52, 53, 55, 99], &[0, 1, 5, 20]);
		
		 // Weird Cases:
		test_pair::<0, 100>(&[], &[50]);
		test_pair::<1, 100>(&[], &[]);
		test_pair::<100, 0>(&[], &[]);
		test_pair::<100, 1>(&[], &[0]);
		test_pair::<1, 1>(&[], &[]);
		test_pair::<0, 0>(&[], &[]);
	}
}