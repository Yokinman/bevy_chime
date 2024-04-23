use std::hash::Hash;
use std::ops::{Deref, DerefMut, RangeTo, RangeFrom, RangeFull};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use bevy_ecs::change_detection::{DetectChanges, Ref, Res};
use bevy_ecs::component::{Component, Tick};
use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::{Query, Resource, World};
use bevy_ecs::query::ArchetypeFilter;
use bevy_ecs::system::{ReadOnlySystemParam, SystemMeta, SystemParam, SystemParamItem};
use bevy_ecs::world::{Mut, unsafe_world_cell::UnsafeWorldCell};
use chime::{Flux, MomentRef, MomentRefMut};
use chime::pred::Prediction;
use crate::node::*;
use crate::comb::*;

/// For [`IntoInput`].
#[derive(Copy, Clone, Default)]
pub struct In<T>(pub T);

/// Resource for passing an event's unique ID to its system parameters. 
#[derive(Resource)]
pub(crate) struct PredSystemInput {
	pub id: Box<dyn std::any::Any + Send + Sync>,
	pub time: Arc<Mutex<Duration>>,
}

/// A hashable unique identifier for a case of prediction.
pub trait PredId:
	Copy + Clone + Eq + Hash + std::fmt::Debug + Send + Sync + 'static
{}

impl<T> PredId for T
where
	T: Copy + Clone + Eq + Hash + std::fmt::Debug + Send + Sync + 'static
{}

/// Linear collections of [`PredParam`] types.
pub trait PredParamVec: PredParam {
	type Head: PredParam;
	type Tail: PredParam;
	
	type Split<'p, K: CombKind>: Iterator<Item = (
		<<Self::Head as PredParam>::Comb<'p> as IntoIterator>::Item,
		PredSubComb<<Self::Tail as PredParam>::Comb<'p>, K>,
	)>;
	
	fn split<K: CombKind>(
		comb: <Self::Comb<'_> as PredCombinator>::IntoKind<K>
	) -> Self::Split<'_, K>;
	
	fn join_id(
		a: <Self::Head as PredParam>::Id,
		b: <Self::Tail as PredParam>::Id,
	) -> Self::Id;
}

impl<A, B> PredParamVec for (A, B)
where
	A: PredParam,
	B: PredParam,
{
	type Head = A;
	type Tail = B;
	
	type Split<'p, K: CombKind> = PredPairCombSplit<
		<<A::Comb<'p> as PredCombinator>::IntoKind<K::Pal> as IntoIterator>::IntoIter,
		B::Comb<'p>,
		K,
	>;
	
	fn split<K: CombKind>(
		comb: <Self::Comb<'_> as PredCombinator>::IntoKind<K>
	) -> Self::Split<'_, K> {
		PredPairCombSplit::new(comb)
	}
	
	fn join_id(
		a: <Self::Head as PredParam>::Id,
		b: <Self::Tail as PredParam>::Id,
	) -> Self::Id {
		(a, b)
	}
}

macro_rules! impl_pred_param_vec_for_array {
	($size:literal) => {
		impl<P> PredParamVec for [P; $size]
		where
			P: PredParam,
			P::Id: Ord,
		{
			type Head = P;
			type Tail = [P; { $size - 1 }];
			
			type Split<'p, K: CombKind>
				= PredArrayCombSplit<P::Comb<'p>, { $size - 1 }, K>;
			
			fn split<K: CombKind>(
				comb: <Self::Comb<'_> as PredCombinator>::IntoKind<K>
			) -> Self::Split<'_, K> {
				PredArrayCombSplit::new(comb)
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
// method to [`PredSubState::outer_iter`] might be worthwhile for convenience.
impl_pred_param_vec_for_array!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

/// Iterator of [`PredSubState::outer_iter`].
pub struct PredSubStateSplitIter<'p, 's, T, P, K>
where
	's: 'p,
	P: PredParamVec,
	K: CombKind,
{
	iter: <P as PredParamVec>::Split<'p, K>,
	branches: NodeWriter<'p, PredNodeBranch<'s, T, P>>,
}

impl<'p, 's, T, P, K> Iterator for PredSubStateSplitIter<'p, 's, T, P, K>
where
	's: 'p,
	P: PredParamVec,
	K: CombKind,
{
	type Item = (
		PredSubStateSplit<'p, 's, T, P::Tail, K>,
		<PredParamItem<'p, P::Head> as PredItem>::Ref,
	);
	fn next(&mut self) -> Option<Self::Item> {
		if let Some((head, tail)) = self.iter.next() {
			let node = &mut self.branches.write((head.id(), PredNode::Blank)).1;
			let sub_state = match tail {
				PredSubComb::Diff(comb) => PredSubStateSplit::Diff(PredSubState::new(comb, node)),
				PredSubComb::Same(comb) => PredSubStateSplit::Same(PredSubState::new(comb, node)),
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

/// Nested state of each [`PredSubState::outer_iter`].
pub enum PredSubStateSplit<'p, 's, T, P, K>
where
	's: 'p,
	P: PredParam,
	K: CombKind,
{
	Diff(PredSubState<'p, 's, T, P, K::Pal>),
	Same(PredSubState<'p, 's, T, P, <<K::Inv as CombKind>::Pal as CombKind>::Inv>),
}

impl<'p, 's, T, P, K> IntoIterator for PredSubStateSplit<'p, 's, T, P, K>
where
	's: 'p,
	P: PredParam,
	K: CombKind,
	PredCombSplit<'p, T, P, K>: Iterator,
	PredSubState<'p, 's, T, P, <K as CombKind>::Pal>:
		IntoIterator<IntoIter = PredComb<'p, T, P, K::Pal>>,
	PredSubState<'p, 's, T, P, <<K::Inv as CombKind>::Pal as CombKind>::Inv>:
		IntoIterator<IntoIter = PredComb<'p, T, P, <<K::Inv as CombKind>::Pal as CombKind>::Inv>>,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = PredCombSplit<'p, T, P, K>;
	fn into_iter(self) -> Self::IntoIter {
		match self {
			Self::Diff(state) => PredCombSplit::Diff(state.into_iter()),
			Self::Same(state) => PredCombSplit::Same(state.into_iter()),
		}
	}
}

/// Shortcut for accessing `PredParam::Comb::Case::Item`.
pub type PredParamItem<'w, P> = <<<P
	as PredParam>::Comb<'w>
	as PredCombinator>::Case
	as PredCombinatorCase>::Item;

/// A set of [`PredItem`] values used to predict & schedule events.
pub trait PredParam {
	/// The equivalent [`bevy_ecs::system::SystemParam`].
	type Param: ReadOnlySystemParam + 'static;
	
	/// Unique identifier for each of [`Self::Param`]'s items.
	type Id: PredId;
	
	/// ...
	type Input: Clone;
	
	/// Creates combinator iterators over [`Self::Param`]'s items.
	type Comb<'w>: PredCombinator<Id=Self::Id>;
	
	/// Produces [`Self::Comb`].
	fn comb<'w>(
		param: &'w SystemParamItem<Self::Param>,
		input: Self::Input,
	) -> Self::Comb<'w>;
}

impl<T, F> PredParam for Query<'_, '_, &T, F>
where
	T: Component,
	F: ArchetypeFilter + 'static,
{
	type Param = Query<'static, 'static, (Ref<'static, T>, Entity), F>;
	type Id = Entity;
	type Input = ();
	type Comb<'w> = QueryComb<'w, T, F>;
	fn comb<'w>(
		param: &'w SystemParamItem<Self::Param>,
		_input: Self::Input,
	) -> Self::Comb<'w> {
		QueryComb::new(param)
	}
}

impl<R> PredParam for Res<'_, R>
where
	R: Resource
{
	type Param = Res<'static, R>;
	type Id = ();
	type Input = ();
	type Comb<'w> = ResComb<'w, R>;
	fn comb<'w>(
		param: &'w SystemParamItem<Self::Param>,
		_input: Self::Input,
	) -> Self::Comb<'w> {
		ResComb::new(Res::clone(param))
	}
}

impl PredParam for () {
	type Param = ();
	type Id = ();
	type Input = ();
	type Comb<'w> = EmptyComb;
	fn comb<'w>(
		_param: &'w SystemParamItem<Self::Param>,
		_input: Self::Input,
	) -> Self::Comb<'w> {
		EmptyComb::new()
	}
}

impl<A, B> PredParam for (A, B)
where
	A: PredParam,
	B: PredParam,
{
	type Param = (A::Param, B::Param);
	type Id = (A::Id, B::Id);
	type Input = (A::Input, B::Input);
	type Comb<'w> = PredPairComb<A::Comb<'w>, B::Comb<'w>>;
	fn comb<'w>(
		(a, b): &'w SystemParamItem<Self::Param>,
		(a_in, b_in): Self::Input,
	) -> Self::Comb<'w> {
		PredPairComb::new(A::comb(a, a_in), B::comb(b, b_in))
	}
}

impl<P, const N: usize> PredParam for [P; N]
where
	P: PredParam,
	P::Id: Ord,
{
	type Param = P::Param;
	type Id = [P::Id; N];
	type Input = [P::Input; N];
	type Comb<'w> = PredArrayComb<P::Comb<'w>, N>;
	fn comb<'w>(
		param: &'w SystemParamItem<Self::Param>,
		input: Self::Input,
	) -> Self::Comb<'w> {
		PredArrayComb::new(P::comb(param, input[0].clone())) // !!! Fix input
	}
}

// impl<C, I> PredParam<I> for &C
// where
// 	C: Component
// {
// 	type Param = Query<'static, 'static, (Ref<'static, C>, Entity)>;
// 	type Id = Entity;
// 	type Comb<'w> = QueryComb<'w, C, ()>;
// 	fn comb<'w>(param: &'w SystemParamItem<Self::Param>) -> Self::Comb<'w> {
// 		QueryComb::new(param)
// 	}
// }

impl<I> PredParam for WithId<I>
where
	I: IntoIterator + Clone,
	I::Item: PredId,
{
	type Param = ();
	type Input = I;
	type Id = I::Item;
	type Comb<'w> = PredIdComb<I>;
	fn comb<'w>(
		_param: &'w SystemParamItem<Self::Param>,
		input: Self::Input,
	) -> Self::Comb<'w> {
		PredIdComb::new(input)
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

/// ...
#[derive(Clone)]
pub struct WithId<I> {
	inner: I,
}

impl<I> PredItem for WithId<I>
where
	I: PredId
{
	type Inner = I;
	type Ref = I;
	fn into_ref(item: Self) -> Self::Ref {
		item.inner
	}
	fn is_updated(_item: &Self) -> bool {
		false
	}
}

/// ...
pub trait PredStateMisc: Clone {
	type Item: PredId;
	type MiscIter: Iterator<Item = Self::Item>;
	fn from_misc(inner: Box<[Self::Item]>) -> Self;
	fn into_misc_iter(self) -> Self::MiscIter;
}

impl PredStateMisc for () {
	type Item = ();
	type MiscIter = std::iter::Once<()>;
	fn from_misc(_inner: Box<[Self::Item]>) -> Self {}
	fn into_misc_iter(self) -> Self::MiscIter {
		std::iter::once(())
	}
}

/// Collects predictions from "when" systems for later compilation. More general
/// form of [`PredState`] for stepping through combinators layer-wise.
pub struct PredSubState<'p, 's, T, P, K>
where
	's: 'p,
	P: PredParam,
	K: CombKind,
{
	pub(crate) comb: <P::Comb<'p> as PredCombinator>::IntoKind<K>,
	pub(crate) node: &'p mut PredNode<'s, T, P>,
}

impl<'p, 's, T, P, K> PredSubState<'p, 's, T, P, K>
where
	's: 'p,
	P: PredParam,
	K: CombKind,
{
	fn new(
		comb: <P::Comb<'p> as PredCombinator>::IntoKind<K>,
		node: &'p mut PredNode<'s, T, P>,
	) -> Self {
		Self { comb, node }
	}
}

impl<'p, 's, T, P, K> PredSubState<'p, 's, T, P, K>
where
	's: 'p,
	T: Prediction + Clone,
	P: PredParam,
	K: CombKind,
{
	/// Sets all updated cases to the given times.
	pub fn set(self, pred: T) {
		let mut iter = self.into_iter();
		if let Some((first, ..)) = iter.next() {
			for (case, ..) in iter {
				case.set(pred.clone());
			}
			first.set(pred);
		}
	}
}

impl<'p, 's, T, P, K> PredSubState<'p, 's, T, P, K>
where
	's: 'p,
	P: PredParamVec,
	K: CombKind,
{
	pub fn outer_iter(self) -> PredSubStateSplitIter<'p, 's, T, P, K> {
		let PredSubState { comb, node } = self;
		let iter = P::split(comb);
		let capacity = 4 * iter.size_hint().0.max(1);
		PredSubStateSplitIter {
			iter,
			branches: node.init_branches(capacity),
		}
	}
}

impl<'p, 's, T, P, K> IntoIterator for PredSubState<'p, 's, T, P, K>
where
	's: 'p,
	P: PredParam,
	K: CombKind,
	PredComb<'p, T, P, K>: Iterator,
{
	type Item = <Self::IntoIter as IntoIterator>::Item;
	type IntoIter = PredComb<'p, T, P, K>;
	fn into_iter(self) -> Self::IntoIter {
		PredComb::new(self)
	}
}

/// Collects predictions from "when" systems for later compilation.
pub struct PredState<'p, 's, T, P>
where
	's: 'p,
	P: PredParam,
{
	inner: PredSubState<'p, 's, T, P, CombAnyTrue>,
}

impl<'p, 's, T, P> PredState<'p, 's, T, P>
where
	's: 'p,
	P: PredParam,
{
	pub(crate) fn new(
		comb: <P::Comb<'p> as PredCombinator>::IntoKind<CombAnyTrue>,
		node: &'p mut PredNode<'s, T, P>,
	) -> Self {
		Self {
			inner: PredSubState::new(comb, node),
		}
	}
}

impl<'p, 's, T, P> PredState<'p, 's, T, P>
where
	's: 'p,
	P: PredParamVec,
{
	pub fn outer_iter(self) -> PredSubStateSplitIter<'p, 's, T, P, CombAnyTrue> {
		self.inner.outer_iter()
	}
}

impl<'p, 's, T, P> IntoIterator for PredState<'p, 's, T, P>
where
	's: 'p,
	P: PredParam,
	PredSubState<'p, 's, T, P, CombAnyTrue>: IntoIterator,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = <PredSubState<'p, 's, T, P, CombAnyTrue> as IntoIterator>::IntoIter;
	fn into_iter(self) -> Self::IntoIter {
		self.inner.into_iter()
	}
}

/// A scheduled case of prediction, used in [`crate::PredState`].
pub struct PredStateCase<I, T> {
	id: I,
	times: Option<T>,
}

impl<I: PredId, T> PredStateCase<I, T> {
	pub fn new(id: I) -> Self {
		Self {
			id,
			times: None,
		}
	}
	
	pub(crate) fn into_parts(self) -> (I, Option<T>) {
		(self.id, self.times)
	}
}

impl<I, T> PredStateCase<I, T>
where
	I: PredId,
	T: Prediction,
{
	pub fn set(&mut self, pred: T) {
		self.times = Some(pred);
	}
}

/// A one-way node that either stores an arbitrary amount of data or branches
/// into sub-nodes.
pub enum PredNode<'s, T: 's, P: PredParam + 's> {
	Blank,
	Data(Node<PredStateCase<P::Id, T>>),
	Branches(Box<dyn PredNodeBranches<'s, T, P> + 's>),
}

impl<'s, T, P> PredNode<'s, T, P>
where
	T: 's,
	P: PredParam + 's
{
	pub fn init_data(&mut self, cap: usize) -> NodeWriter<PredStateCase<P::Id, T>> {
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
	
	fn init_branches(&mut self, cap: usize) -> NodeWriter<PredNodeBranch<'s, T, P>>
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

impl<'s, T, P: PredParam> IntoIterator for PredNode<'s, T, P> {
	type Item = PredStateCase<P::Id, T>;
	type IntoIter = PredNodeIter<'s, T, P>;
	fn into_iter(self) -> Self::IntoIter {
		match self {
			Self::Blank => PredNodeIter::Blank,
			Self::Data(node) => PredNodeIter::Data(node.into_iter()),
			Self::Branches(mut branches) => PredNodeIter::Branches(branches.into_branch_iter()),
		}
	}
}

/// Iterator of [`PredNode`]'s items.
pub enum PredNodeIter<'s, T, P: PredParam> {
	Blank,
	Data(NodeIter<PredStateCase<P::Id, T>>),
	Branches(Box<dyn PredNodeBranchesIterator<'s, T, P> + 's>),
}

impl<T, P: PredParam> Iterator for PredNodeIter<'_, T, P> {
	type Item = PredStateCase<P::Id, T>;
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
type PredNodeBranch<'s, T, P> = (
	<<P as PredParamVec>::Head as PredParam>::Id,
	PredNode<'s, T, <P as PredParamVec>::Tail>,
);

/// Used to define a trait object for dynamic branching in [`PredNode`], as not
/// all [`PredParam`] types implement [`PredParamVec`].
/// 
/// ??? To avoid dynamic dispatch: could move all `PredParamVec` items into
/// `PredParam` and implement empty defaults for scalar types. However, this
/// would only support a subset of arrays instead of all sizes, which feels
/// like an unnecessary constraint. Specialization would probably help here.
pub trait PredNodeBranches<'s, T, P: PredParam> {
	fn as_writer<'n>(&'n mut self) -> NodeWriter<'n, PredNodeBranch<'s, T, P>>
	where
		P: PredParamVec;
	
	fn into_branch_iter(&mut self) -> Box<dyn PredNodeBranchesIterator<'s, T, P> + 's>;
}

impl<'s, T, P> PredNodeBranches<'s, T, P> for Node<PredNodeBranch<'s, T, P>>
where
	T: 's,
	P: PredParamVec + 's,
{
	fn as_writer<'n>(&'n mut self) -> NodeWriter<'n, PredNodeBranch<'s, T, P>>
	where
		P: PredParamVec
	{
		NodeWriter::new(self)
	}
	
	fn into_branch_iter(&mut self) -> Box<dyn PredNodeBranchesIterator<'s, T, P> + 's> {
		Box::new(PredNodeBranchesIter {
			node_iter: std::mem::take(self).into_iter(),
			branch_id: None,
			branch_iter: PredNodeIter::Blank,
		})
	}
}

/// Specific type of [`PredNodeBranchesIterator`] trait objects.
pub struct PredNodeBranchesIter<'s, T, P: PredParamVec> {
	node_iter: NodeIter<PredNodeBranch<'s, T, P>>,
	branch_id: Option<<P::Head as PredParam>::Id>,
	branch_iter: PredNodeIter<'s, T, P::Tail>,
}

impl<'s, T, P> Iterator for PredNodeBranchesIter<'s, T, P>
where
	P: PredParamVec
{
	type Item = PredStateCase<P::Id, T>;
	fn next(&mut self) -> Option<Self::Item> {
		if let Some(case) = self.branch_iter.next() {
			Some(PredStateCase {
				id: P::join_id(self.branch_id.unwrap(), case.id),
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
pub trait PredNodeBranchesIterator<'s, T, P: PredParam>:
	Iterator<Item = PredStateCase<P::Id, T>>
{}

impl<'s, T, P> PredNodeBranchesIterator<'s, T, P> for PredNodeBranchesIter<'s, T, P>
where
	P: PredParamVec
{}

/// Types that can be used to query for a specific entity.
pub trait PredFetchData {
	type Id: PredId;
	type Output<'w>;
	unsafe fn get_inner(world: UnsafeWorldCell, id: Self::Id) -> Self::Output<'_>;
	// !!! Could take a dynamically-dispatched ID and attempt downcasting
	// manually. Return an `Option<Self::Output>` for whether it worked. This
	// would allow for query types that accept multiple IDs; support `() -> ()`.
}

impl PredFetchData for () {
	type Id = ();
	type Output<'w> = ();
	unsafe fn get_inner(_world: UnsafeWorldCell, _id: Self::Id) -> Self::Output<'_> {}
}

impl<C: Component> PredFetchData for &C {
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

impl<C: Component> PredFetchData for &mut C {
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

impl<C: Component, const N: usize> PredFetchData for [&C; N] {
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

impl<C: Component, const N: usize> PredFetchData for [&mut C; N] {
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

impl<A: PredFetchData, B: PredFetchData> PredFetchData for (A, B) {
	type Id = (A::Id, B::Id);
	type Output<'w> = (A::Output<'w>, B::Output<'w>);
	unsafe fn get_inner(world: UnsafeWorldCell, (a, b): Self::Id) -> Self::Output<'_> {
		(A::get_inner(world, a), B::get_inner(world, b))
	}
}

impl<I> PredFetchData for WithId<I>
where
	I: PredId
{
	type Id = I;
	type Output<'w> = I;
	unsafe fn get_inner(_world: UnsafeWorldCell, id: Self::Id) -> Self::Output<'_> {
		// SAFETY: This doesn't access the world.
		id
	}
}

/// Prediction data fed as a parameter to an event's systems.
pub struct PredFetch<'world, D: PredFetchData> {
    inner: D::Output<'world>,
	time: Duration,
}

impl<'w, D: PredFetchData> PredFetch<'w, D> {
	pub fn get_inner(self) -> D::Output<'w> {
		self.inner
	}
}

impl<'w, D: PredFetchData> PredFetch<'w, D>
where
	D::Output<'w>: Deref,
	<D::Output<'w> as Deref>::Target: Flux + Clone,
{
	/// ...
	pub fn moment(&self)
		-> MomentRef<<<D::Output<'w> as Deref>::Target as Flux>::Moment>
	{
		self.inner.at(self.time)
	}
	
	/// ...
	pub fn moment_mut(&mut self)
		-> MomentRefMut<<<D::Output<'w> as Deref>::Target as Flux>::Moment>
	where
		D::Output<'w>: DerefMut,
	{
		self.inner.at_mut(self.time)
	}
}

unsafe impl<D: PredFetchData> SystemParam for PredFetch<'_, D> {
	type State = (D::Id, Arc<Mutex<Duration>>);
	type Item<'world, 'state> = PredFetch<'world, D>;
	fn init_state(world: &mut World, system_meta: &mut SystemMeta) -> Self::State {
		// !!! Check for component access overlap. This isn't safe right now.
		if let Some(PredSystemInput { id, time, .. }) = world
			.get_resource::<PredSystemInput>()
		{
			if let Some(id) = id.downcast_ref::<D::Id>() {
				(*id, Arc::clone(time))
			} else {
				panic!(
					"!!! parameter is for wrong ID type. got {:?}",
					std::any::type_name::<D::Id>()
				);
			}
		} else {
			panic!("!!! {:?} is not a Chime event system, it can't use this parameter type", system_meta.name());
		}
	}
	// fn new_archetype(_state: &mut Self::State, _archetype: &Archetype, _system_meta: &mut SystemMeta) {
	// 	todo!()
	// }
	unsafe fn get_param<'world, 'state>((id, time): &'state mut Self::State, _system_meta: &SystemMeta, world: UnsafeWorldCell<'world>, _change_tick: Tick) -> Self::Item<'world, 'state> {
		PredFetch {
			inner: D::get_inner(world, *id),
			time: *time.lock().expect("should be available"),
		}
	}
}

/// ...
/// 
/// ## Types of Input
/// 
/// [`IntoIterator`] ([`In`])
/// ```text
/// In<A> -> B
/// where
///     A: IntoIterator,
///     B: FromIterator<A::Item>,
/// ```
/// 
/// Tuples (up to size 4, including unit)
/// ```text
/// (A,) -> (B,)
/// where
///     A: IntoInput<B>
/// ```
/// 
/// Arrays (any length)
/// ```text
/// [A; N] -> [B; N]
/// where
///     A: IntoInput<B>
/// ```
/// 
/// [`Default`] ([`RangeFull`])
/// ```text
/// .. -> T where T: Default
/// ..(A,) -> (*, B) where A: IntoInput<B>, *: Default
/// (A,).. -> (B, *) where A: IntoInput<B>, *: Default
/// ```
pub trait IntoInput<I> {
	fn into_input(self) -> I;
}

impl<A, B> IntoInput<B> for In<A>
where
	A: IntoIterator,
	B: FromIterator<A::Item>,
{
	fn into_input(self) -> B {
		B::from_iter(self.0)
	}
}

impl<T> IntoInput<T> for RangeFull
where
	T: Default
{
	fn into_input(self) -> T {
		T::default()
	}
}

impl IntoInput<()> for () {
	fn into_input(self) {}
}

macro_rules! impl_into_input_for_tuples {
	($(:$bx:ident,)+ $a0:ident : $b0:ident $(, $a:ident : $b:ident)*) => {
		 // Range Head Defaults:
		impl<$($bx,)+ $a0, $b0, $($a, $b,)*> IntoInput<($($bx,)+ $b0, $($b,)*)>
			for RangeTo<($a0, $($a,)*)>
		where
			$($bx: Default,)+
			$a0: IntoInput<$b0>,
			$($a: IntoInput<$b>,)*
		{
			fn into_input(self) -> ($($bx,)+ $b0, $($b,)*) {
				#[allow(non_snake_case)]
				let ($a0, $($a,)*) = self.end;
				($($bx::default(),)+ $a0.into_input(), $($a.into_input(),)*)
			}
		}
		
		 // Range Tail Defaults:
		impl<$($a, $b,)* $a0, $b0, $($bx,)+> IntoInput<($($b,)* $b0, $($bx,)+)>
			for RangeFrom<($($a,)* $a0,)>
		where
			$($a: IntoInput<$b>,)*
			$a0: IntoInput<$b0>,
			$($bx: Default,)+
		{
			fn into_input(self) -> ($($b,)* $b0, $($bx,)+) {
				#[allow(non_snake_case)]
				let ($($a,)* $a0,) = self.start;
				($($a.into_input(),)* $a0.into_input(), $($bx::default(),)+)
			}
		}
		
		impl_into_input_for_tuples!{$(:$bx,)+ :$b0 $(, $a:$b)*}
	};
	($a0:ident : $b0:ident $(, $a:ident : $b:ident)*) => {
		impl<$a0, $b0, $($a, $b,)*> IntoInput<($b0, $($b,)*)>
			for ($a0, $($a,)*)
		where
			$a0: IntoInput<$b0>,
			$($a: IntoInput<$b>,)*
		{
			fn into_input(self) -> ($b0, $($b,)*) {
				#[allow(non_snake_case)]
				let ($a0, $($a,)*) = self;
				($a0.into_input(), $($a.into_input(),)*)
			}
		}
		
		impl_into_input_for_tuples!{:$b0 $(, $a:$b)*}
		impl_into_input_for_tuples!{$($a:$b),*}
	};
	($(:$b:ident),*) => {};
}

impl_into_input_for_tuples!{
	A0:B0, A1:B1, A2:B2, A3:B3
}

impl<A, B, const N: usize> IntoInput<[B; N]> for [A; N]
where
	A: IntoInput<B>,
{
	fn into_input(self) -> [B; N] {
		self.map(A::into_input)
	}
}

#[cfg(test)]
mod testing {
	use super::*;
	use chime::pred::DynPred;
	
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
		for<'w, 's, 'a, 'w1, 's1, 'a1>
			<(Query<'w, 's, &'a Test>, Query<'w1, 's1, &'a1 TestB>)
				as PredParam>::Input:
					Default + IntoInput<<(Query<'w, 's, &'a Test>, Query<'w1, 's1, &'a1 TestB>)
				as PredParam>::Input> + Send + Sync + 'static,
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
			state: PredState<DynPred, (Query<&Test>, Query<&TestB>)>,
			a_query: Query<&Test>,
			b_query: Query<&TestB>,
			mut index: system::Local<usize>,
		| {
			let mut iter = state.into_iter();
			match *index {
				0 => { // Full
					if A > 1 && B > 1 {
						assert_eq!(iter.size_hint(), (B, Some(A*B)));
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
						assert_eq!(iter.size_hint(), (B, Some(A*B)));
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
		
		 // Setup [`PredSubState::outer_iter`] Testing:
		let update_vec = update_list.to_vec();
		let b_update_vec = b_update_list.to_vec();
		app.add_chime_events((move |
			state: PredState<DynPred, (Query<&Test>, Query<&TestB>)>,
			a_query: Query<Ref<Test>>,
			b_query: Query<Ref<TestB>>,
			mut index: system::Local<usize>,
		| {
			let mut iter = state.outer_iter();
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
				as PredCombinator>::Case
				as PredCombinatorCase>::Item
				as PredItem>::Ref:
					IntoIterator,
		for<'w, 's, 'a, 'b>
			<<<<<<[Query<'w, 's, &'a Test>; R]
				as PredParamVec>::Tail
				as PredParam>::Comb<'b>
				as PredCombinator>::Case
				as PredCombinatorCase>::Item
				as PredItem>::Ref
				as IntoIterator>::Item:
					Deref<Target = Test>,
		for<'w, 's, 'a> <[Query<'w, 's, &'a Test>; R] as PredParam>::Input:
			Default + IntoInput<<[Query<'w, 's, &'a Test>; R] as PredParam>::Input> + Send + Sync + 'static,
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
			state: PredState<DynPred, [Query<&Test>; R]>,
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
		
		 // Setup [`PredSubState::outer_iter`] Testing:
		let update_vec = update_list.to_vec();
		app.add_chime_events((move |
			state: PredState<DynPred, [Query<&Test>; R]>,
			query: Query<Ref<Test>>,
			mut index: system::Local<usize>,
		| {
			let mut iter = state.outer_iter();
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