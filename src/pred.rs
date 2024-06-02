use std::hash::Hash;
use std::ops::{Deref, DerefMut, RangeTo, RangeFrom, RangeFull};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use bevy_ecs::change_detection::{DetectChanges, Ref, Res};
use bevy_ecs::component::{Component, Tick};
use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::{Query, Resource, World};
use bevy_ecs::query::{ArchetypeFilter, QueryData, ReadOnlyQueryData, WorldQuery};
use bevy_ecs::system::{ReadOnlySystemParam, SystemMeta, SystemParam, SystemParamItem};
use bevy_ecs::world::{Mut, unsafe_world_cell::UnsafeWorldCell};
use chime::{Flux, MomentRef, MomentRefMut};
use chime::pred::Prediction;
use crate::node::*;
use crate::comb::*;

/// For input to [`crate::AddChimeEvent::add_chime_events`].
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

/// Shortcut for accessing `PredParam::Comb::Case::Item`.
pub type PredParamItem<P> = <P as PredCombinator>::Item_;

/// A set of [`PredItem`] values used to predict & schedule events.
pub trait PredCombinator: Clone + IntoIterator<Item=Self::Case> {
	/// The equivalent [`bevy_ecs::system::SystemParam`].
	type Param: ReadOnlySystemParam + 'static;
	
	/// Unique identifier for each of [`Self::Param`]'s items.
	type Id: PredId;
	type Item_: PredItem;
	type Case: PredCombinatorCase<Id=Self::Id, Item=Self::Item_>;
	
	/// ...
	type Input: Clone;
	
	/// Creates combinator iterators over [`Self::Param`]'s items.
	type Comb<K: CombKind>:
		PredCombinator<Case=Self::Case, Id=Self::Id>;
	
	/// Produces [`Self::Comb`].
	fn comb<K: CombKind>(
		param: &'static SystemParamItem<Self::Param>,
		kind: K,
		input: Self::Input,
	) -> Self::Comb<K>;
}

mod _pred_combinator_impls {
	use super::*;

	impl<'w, T, F, Kind> PredCombinator for QueryComb<'w, T, F, Kind>
	where
		T: PredParamQueryData,
		F: ArchetypeFilter + 'static,
		T::Item<'w>: PredItem,
		<T::ItemRef as WorldQuery>::Item<'w>: PredItemRef<Item = T::Item<'w>>,
		Kind: CombKind,
	{
		type Param = Query<'static, 'static, (T::ItemRef, Entity), F>;
		type Id = Entity;
		type Item_ = Fetch<'w, T, F>;
		type Case = PredCombCase<Self::Item_, Self::Id>;
		type Input = ();
		type Comb<K: CombKind> = QueryComb<'w, T, F, K>;
		fn comb<K: CombKind>(
			param: &'static SystemParamItem<Self::Param>,
			kind: K,
			_input: Self::Input,
		) -> Self::Comb<K> {
			QueryComb::new(param, kind)
		}
	}
	
	impl<'w, R, Kind> PredCombinator for ResComb<'w, R, Kind>
	where
		R: Resource,
		Kind: CombKind,
	{
		type Param = Res<'static, R>;
		type Id = ();
		type Item_ = Res<'w, R>;
		type Case = PredCombCase<Self::Item_, Self::Id>;
		type Input = ();
		type Comb<K: CombKind> = ResComb<'w, R, K>;
		fn comb<K: CombKind>(
			param: &'static SystemParamItem<Self::Param>,
			kind: K,
			_input: Self::Input,
		) -> Self::Comb<K> {
			ResComb::new(Res::clone(param), kind)
		}
	}
	
	impl<Kind> PredCombinator for EmptyComb<Kind>
	where
		Kind: CombKind,
	{
		type Param = ();
		type Id = ();
		type Item_ = ();
		type Case = PredCombCase<Self::Item_, Self::Id>;
		type Input = ();
		type Comb<K: CombKind> = EmptyComb<K>;
		fn comb<K: CombKind>(
			_param: &'static SystemParamItem<Self::Param>,
			kind: K,
			_input: Self::Input,
		) -> Self::Comb<K> {
			EmptyComb::new(kind)
		}
	}
	
	impl<A, Kind> PredCombinator for PredSingleComb<A, Kind>
	where
		A: PredCombinator,
		Kind: CombKind,
	{
		type Param = A::Param;
		type Id = (A::Id,);
		type Item_ = (A::Item_,);
		type Case = (A::Case,);
		type Input = (A::Input,);
		type Comb<K: CombKind> = PredSingleComb<A, K>;
		fn comb<K: CombKind>(
			param: &'static SystemParamItem<Self::Param>,
			kind: K,
			(input,): Self::Input,
		) -> Self::Comb<K> {
			PredSingleComb::new(A::comb(param, kind, input))
		}
	}
	
	impl<A, B, Kind> PredCombinator for PredPairComb<A, B, Kind>
	where
		A: PredCombinator,
		B: PredCombinator,
		Kind: CombKind,
	{
		type Param = (A::Param, B::Param);
		type Id = (A::Id, B::Id);
		type Item_ = (A::Item_, B::Item_);
		type Case = (A::Case, B::Case);
		type Input = (A::Input, B::Input);
		type Comb<K: CombKind> = PredPairComb<A, B, K>;
		fn comb<K: CombKind>(
			(a, b): &'static SystemParamItem<Self::Param>,
			kind: K,
			(a_in, b_in): Self::Input,
		) -> Self::Comb<K> {
			PredPairComb::new(
				A::comb(a, kind, a_in.clone()),
				B::comb(b, kind.pal(), b_in.clone()),
				A::comb(a, kind.inv(), a_in),
				B::comb(b, kind.inv().pal().inv(), b_in),
				kind,
			)
		}
	}
	
	impl<P, const N: usize, Kind> PredCombinator for PredArrayComb<P, N, Kind>
	where
		P: PredCombinator,
		P::Id: Ord,
		Kind: CombKind,
	{
		type Param = P::Param;
		type Id = [P::Id; N];
		type Item_ = [P::Item_; N];
		type Case = [P::Case; N];
		type Input = [P::Input; N];
		type Comb<K: CombKind> = PredArrayComb<P, N, K>;
		fn comb<K: CombKind>(
			param: &'static SystemParamItem<Self::Param>,
			kind: K,
			input: Self::Input,
		) -> Self::Comb<K> {
			PredArrayComb::new(
				P::comb(param, CombBranch::A(kind.pal()), input[0].clone()),
				P::comb(param, CombBranch::B(kind), input[0].clone()),
				kind,
			) // !!! Fix input
		}
	}
	
	impl<I> PredCombinator for PredIdComb<I>
	where
		I: IntoIterator + Clone,
		I::Item: PredId,
	{
		type Param = ();
		type Input = I;
		type Id = I::Item;
		type Item_ = WithId<I::Item>;
		type Case = PredCombCase<Self::Item_, Self::Id>;
		type Comb<K: CombKind> = PredIdComb<I>;
		fn comb<K: CombKind>(
			_param: &'static SystemParamItem<Self::Param>,
			_kind: K,
			input: Self::Input,
		) -> Self::Comb<K> {
			PredIdComb::new(input)
		}
	}
	
	impl<T> PredCombinator for Misc<T>
	where
		T: PredCombinator,
	{
		type Param = T::Param;
		type Input = T::Input;
		type Id = T::Id;
		type Item_ = Misc<T::Item_>;
		type Case = Misc<T::Case>;
		type Comb<K: CombKind> = Misc<T::Comb<K>>;
		fn comb<K: CombKind>(
			param: &'static SystemParamItem<Self::Param>,
			kind: K,
			input: Self::Input,
		) -> Self::Comb<K> {
			Misc(T::comb(param, kind, input))
		}
	}
	
	impl<T> PredCombinatorCase for Misc<T>
	where
		T: PredCombinatorCase,
	{
		type Id = T::Id;
		type Item = Misc<T::Item>;
		fn is_diff(&self) -> bool {
			false
		}
		fn into_parts(self) -> (Self::Item, Self::Id) {
			let Misc(inner) = self;
			let (item, id) = T::into_parts(inner);
			(Misc(item), id)
		}
	}
}

/// `PredParam` wrapper for stripping the updated state.
#[derive(Clone)]
pub struct Misc<T>(pub T);

impl<T> PredItem for Misc<T>
where
	T: PredItem,
{
	fn clone(&self) -> Self {
		let Misc(inner) = self;
		Misc(inner.clone())
	}
}

impl<T> IntoIterator for Misc<T>
where
	T: IntoIterator,
{
	type Item = Misc<T::Item>;
	type IntoIter = MiscIter<T::IntoIter>;
	fn into_iter(self) -> Self::IntoIter {
		let Misc(inner) = self;
		MiscIter(inner.into_iter())
	}
}

/// ...
pub struct MiscIter<T>(T);

impl<T> Iterator for MiscIter<T>
where
	T: Iterator,
{
	type Item = Misc<T::Item>;
	fn next(&mut self) -> Option<Self::Item> {
		let MiscIter(inner) = self;
		inner.next().map(Misc)
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		let MiscIter(inner) = self;
		inner.size_hint()
	}
}

/// ...
pub trait PredParamQueryData: ReadOnlyQueryData {
	type ItemRef: ReadOnlyQueryData + 'static;
}

impl<'a, T> PredParamQueryData for &'a T
where
	T: Component,
{
	type ItemRef = Ref<'static, T>;
}

impl<A, B> PredParamQueryData for (A, B)
where
	A: PredParamQueryData,
	B: PredParamQueryData,
{
	type ItemRef = (A::ItemRef, B::ItemRef);
}

/// Unused - may replace the output of `QueryComb` if the concept of
/// case-by-case prediction closures is implemented.
#[derive(Debug)]
pub struct Fetch<'w, D: QueryData, F = ()> {
	inner: D::Item<'w>,
	_filter: std::marker::PhantomData<F>,
}

impl<'w, D: QueryData, F> Fetch<'w, D, F> {
	pub(crate) fn new(inner: D::Item<'w>) -> Self {
		Self {
			inner,
			_filter: std::marker::PhantomData,
		}
	}
	
	pub fn into_inner(self) -> D::Item<'w> {
		self.inner
	}
}

impl<'w, D: QueryData, F> Deref for Fetch<'w, D, F> {
	type Target = D::Item<'w>;
	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

/// Case of prediction.
pub trait PredItem {
	fn clone(&self) -> Self;
}

mod _pred_item_impls {
	use super::*;
	
	impl PredItem for () {
		fn clone(&self) -> Self {}
	}
	
	impl<T> PredItem for &T {
		fn clone(&self) -> Self {
			self
		}
	}
	
	impl<'w, D, F> PredItem for Fetch<'w, D, F>
	where
		D: QueryData,
		D::Item<'w>: PredItem,
	{
		fn clone(&self) -> Self {
			Fetch::new(self.inner.clone())
		}
	}
	
	impl<T: Resource> PredItem for Res<'_, T> {
		fn clone(&self) -> Self {
			Res::clone(self) // GGGRRAAAAAAHHHHH!!!!!!!!!
		}
	}
	
	impl<A,> PredItem for (A,)
	where
		A: PredItem,
	{
		fn clone(&self) -> Self {
			let (a,) = self;
			(a.clone(),)
		}
	}
	
	impl<A, B> PredItem for (A, B)
	where
		A: PredItem,
		B: PredItem,
	{
		fn clone(&self) -> Self {
			let (a, b) = self;
			(a.clone(), b.clone())
		}
	}
	
	impl<T, const N: usize> PredItem for [T; N]
	where
		T: PredItem
	{
		fn clone(&self) -> Self {
			self.each_ref()
				.map(T::clone)
		}
	}
	
	impl<I> PredItem for WithId<I>
	where
		I: PredId
	{
		fn clone(&self) -> Self {
			*self
		}
	}
}

/// ...
pub trait PredItem2<P: PredCombinator<Item_= Self> + ?Sized>: PredItem {}

mod _pred_item2_impls {
	use super::*;
	
	impl PredItem2<EmptyComb> for () {}
	
	impl<'w, D, F> PredItem2<QueryComb<'w, D, F>> for Fetch<'w, D, F>
	where
		D: PredParamQueryData,
		F: ArchetypeFilter + 'static,
		D::Item<'w>: PredItem,
		<D::ItemRef as WorldQuery>::Item<'w>: PredItemRef<Item = D::Item<'w>>,
	{}
	
	impl<'w, T: Resource> PredItem2<ResComb<'w, T>> for Res<'w, T> {}
	
	impl<A, P,> PredItem2<PredSingleComb<P>> for (A,)
	where
		A: PredItem2<P>,
		P: PredCombinator<Item_= A>,
	{}
	
	impl<A, B, P, Q,> PredItem2<PredPairComb<P, Q>> for (A, B,)
	where
		A: PredItem2<P>,
		B: PredItem2<Q>,
		P: PredCombinator<Item_= A>,
		Q: PredCombinator<Item_= B>,
	{}
	
	impl<T, P, const N: usize> PredItem2<PredArrayComb<P, N>> for [T; N]
	where
		T: PredItem2<P>,
		P: PredCombinator<Item_= T>,
		P::Id: Ord,
	{}
	
	impl<T> PredItem2<PredIdComb<T>> for WithId<T::Item>
	where
		T: IntoIterator + Clone,
		T::Item : PredId,
	{}
	
	impl<T, I> PredItem2<Misc<T>> for Misc<I>
	where
		T: PredCombinator<Item_= I>,
		I: PredItem2<T>,
	{}
}

/// [`PredItem`] with updated state.
pub trait PredItemRef {
	type Item: PredItem;
	fn into_item(self) -> Self::Item;
	
	/// Whether this item is in need of a prediction update.
	fn is_updated(item: &Self) -> bool;
}

mod _pred_item_ref_impls {
	use super::*;
	
	impl<'w, T: 'static> PredItemRef for Ref<'w, T> {
		type Item = &'w T;
		fn into_item(self) -> Self::Item {
			self.into_inner()
		}
		fn is_updated(item: &Self) -> bool {
			DetectChanges::is_changed(item)
		}
	}
	
	impl<'w, R: Resource> PredItemRef for Res<'w, R> {
		type Item = Self;
		fn into_item(self) -> Self::Item {
			self
		}
		fn is_updated(item: &Self) -> bool {
			DetectChanges::is_changed(item)
		}
	}
	
	impl PredItemRef for () {
		type Item = ();
		fn into_item(self) -> Self::Item {
			self
		}
		fn is_updated(_item: &Self) -> bool {
			true
		}
	}
	
	impl<A, B> PredItemRef for (A, B)
	where
		A: PredItemRef,
		B: PredItemRef,
	{
		type Item = (A::Item, B::Item);
		fn into_item(self) -> Self::Item {
			let (a, b) = self;
			(a.into_item(), b.into_item())
		}
		fn is_updated(item: &Self) -> bool {
			let (a, b) = item;
			A::is_updated(a) || B::is_updated(b)
		}
	}
}

/// ...
#[derive(Copy, Clone)]
pub struct WithId<I>(pub I);

/// ...
pub struct PredSubState2<'p, T, P, K>
where
	P: PredBranch + ?Sized,
	K: CombKind,
{
	pub(crate) comb: P::CombSplit<K>,
	pub(crate) node: &'p mut Node<P::Case<T>>,
	pub(crate) kind: K,
}

impl<'p, T, P, K> PredSubState2<'p, T, P, K>
where
	P: PredBranch,
	K: CombKind,
{
	pub(crate) fn new(
		comb: P::CombSplit<K>,
		node: &'p mut Node<P::Case<T>>,
		kind: K,
	) -> Self {
		Self {
			comb,
			node,
			kind,
		}
	}
}

impl<'p, T, P, K> IntoIterator for PredSubState2<'p, T, P, K>
where
	T: Prediction + 'p,
	P: PredBranch,
	P::Branch: 'p,
	K: CombKind,
	P::Comb<'p, T, K>: Iterator,
{
	type Item = <Self::IntoIter as IntoIterator>::Item;
	type IntoIter = P::Comb<'p, T, K>;
	fn into_iter(self) -> Self::IntoIter {
		P::new_comb(self)
	}
}

/// Collects predictions from "when" systems for later compilation.
pub struct PredState2<'p, T, P>
where
	P: PredBranch,
{
	pub(crate) inner: PredSubState2<'p, T, P, CombAnyTrue>,
}

impl<'p, T, P> PredState2<'p, T, P>
where
	P: PredBranch,
{
	pub(crate) fn new(
		comb: P::CombSplit<CombAnyTrue>,
		node: &'p mut Node<P::Case<T>>,
	) -> Self {
		Self {
			inner: PredSubState2::new(comb, node, CombAnyTrue),
		}
	}
}

impl<'p, T, P> IntoIterator for PredState2<'p, T, P>
where
	P: PredBranch,
	PredSubState2<'p, T, P, CombAnyTrue>: IntoIterator,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = <PredSubState2<'p, T, P, CombAnyTrue> as IntoIterator>::IntoIter;
	fn into_iter(self) -> Self::IntoIter {
		self.inner.into_iter()
	}
}

/// A scheduled case of prediction, used in [`crate::PredState`].
pub struct PredStateCase<I, T> {
	id: I,
	pred: Option<T>,
}

impl<I, T> PredStateCase<I, T> {
	pub(crate) fn new(id: I) -> Self {
		Self {
			id,
			pred: None,
		}
	}
	
	pub(crate) fn into_parts(self) -> (I, Option<T>) {
		(self.id, self.pred)
	}
}

impl<I, T> PredStateCase<I, T>
where
	I: PredId,
	T: Prediction,
{
	pub fn set(&mut self, pred: T) {
		self.pred = Some(pred);
	}
}

/// ...
pub trait PredPermBranch: PredBranch /*+ std::ops::Index<usize>*/ {
	type Output;
	
	fn depth() -> usize;
	
	fn sort_unstable(&mut self)
	where
		Self: Ord
	{
		todo!()
	}
	
	fn outer_skip<K: CombKind>(
		comb: &mut <Self as PredBranch>::CombSplit<K>,
		index: [usize; 2],
	);
}

mod _pred_perm_branch_impls {
	use super::*;
	
	impl<T, const N: usize> PredPermBranch for Single<PredArrayComb<T, N>>
	where
		T: PredCombinator,
		T::Id: Ord,
	{
		type Output = T;
		
		fn depth() -> usize {
			N
		}
	
		fn outer_skip<K: CombKind>(
			comb: &mut <Self as PredBranch>::CombSplit<K>,
			index: [usize; 2],
		) {
			comb.a_index += index[0];
			comb.b_index += index[1];
		}
	}
	
	impl<A, const N: usize, B> PredPermBranch for NestedPerm<PredArrayComb<A, N>, B>
	where
		A: PredCombinator,
		A::Id: Ord,
		B: PredPermBranch<Output = A>,
	{
		type Output = A;
		
		fn depth() -> usize {
			N + B::depth()
		}
	
		fn outer_skip<K: CombKind>(
			(outer, _inner): &mut <Self as PredBranch>::CombSplit<K>,
			index: [usize; 2],
		) {
			outer.a_index += index[0];
			outer.b_index += index[1];
			// B::outer_skip(&mut inner[0], index);
			// B::outer_skip(&mut inner[1], index);
		}
	}
}

/// ...
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default)]
pub struct Single<T>(pub T);

/// ...
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default)]
pub struct Nested<A, B>(pub A, pub B);

/// ...
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default)]
pub struct NestedPerm<A, B>(pub A, pub B);

/// ...
pub trait PredBranch {
	type Param: PredCombinator;
	type Branch: PredBranch;
	type Case<T>: PredNodeCase<Id = <Self::Param as PredCombinator>::Id>;
	type AllParams: PredCombinator;
	type Id: PredId;
	type Input: Clone;
	type CombSplit<K: CombKind>: Clone;
	
	type Item<'p, T, K>
	where
		T: Prediction + 'p,
		K: CombKind,
		Self::Branch: 'p;
	
	type Comb<'p, T, K>: Iterator<Item = (
		Self::Item<'p, T, K>,
		PredParamItem<Self::Param>,
	)>
	where
		T: Prediction + 'p,
		K: CombKind,
		Self::Branch: 'p;
	
	type CaseIter<T>: Iterator<Item = PredStateCase<Self::Id, T>>;
	
	fn comb_split<K: CombKind>(
		params: &'static SystemParamItem<<Self::AllParams as PredCombinator>::Param>,
		input: Self::Input,
		kind: K,
	) -> Self::CombSplit<K>;
	
	fn case_iter<T>(case: Self::Case<T>) -> Self::CaseIter<T>;
	
	fn new_comb<'p, T, K>(state: PredSubState2<'p, T, Self, K>) -> Self::Comb<'p, T, K>
	where
		T: Prediction + 'p,
		K: CombKind,
		Self::Branch: 'p;
}

mod _pred_branch_impls {
	use super::*;
	
	impl<A> PredBranch for Single<A>
	where
		A: PredCombinator
	{
		type Param = A;
		type Branch = Single<EmptyComb>;
		type Case<T> = PredStateCase<A::Id, T>;
		type AllParams = A;
		type Id = A::Id;
		type Input = Single<A::Input>;
		type CombSplit<K: CombKind> = A::Comb<K>;
		
		type Item<'p, T, K> = &'p mut Self::Case<T>
		where
			T: Prediction + 'p,
			K: CombKind,
			Self::Branch: 'p;
		
		type Comb<'p, T, K> = SinglePredComb2<'p, T, Self::Param, K>
		where
			T: Prediction + 'p,
			K: CombKind,
			Self::Branch: 'p;
		
		type CaseIter<T> = std::iter::Once<PredStateCase<A::Id, T>>;
		
		fn comb_split<K: CombKind>(
			params: &'static SystemParamItem<<Self::AllParams as PredCombinator>::Param>,
			Single(input): Self::Input,
			kind: K,
		) -> Self::CombSplit<K> {
			A::comb(params, kind, input)
		}
		
		fn case_iter<T>(case: Self::Case<T>) -> Self::CaseIter<T> {
			std::iter::once(case)
		}
		
		fn new_comb<'p, T, K>(state: PredSubState2<'p, T, Self, K>) -> Self::Comb<'p, T, K>
		where
			T: Prediction + 'p,
			K: CombKind,
			Self::Branch: 'p,
		{
			SinglePredComb2::new(state)
		}
	}
	
	impl<A, B> PredBranch for Nested<A, B>
	where
		A: PredCombinator,
		B: PredBranch,
	{
		type Param = A;
		type Branch = B;
		type Case<T> = (A::Id, Node<B::Case<T>>);
		type AllParams = PredPairComb<A, B::AllParams>;
		type Id = Nested<A::Id, B::Id>;
		type Input = Nested<A::Input, B::Input>;
		type CombSplit<K: CombKind> = (
			<Self::Param as PredCombinator>::Comb<K::Pal>,
			[B::CombSplit<CombBranch<K::Pal, K>>; 2],
		);
		
		type Item<'p, T, K> = PredSubState2<'p, T, B, CombBranch<K::Pal, K>>
		where
			T: Prediction + 'p,
			K: CombKind,
			Self::Branch: 'p;
		
		type Comb<'p, T, K> = NestedPredComb2<'p, T, A, B, K>
		where
			T: Prediction + 'p,
			K: CombKind,
			Self::Branch: 'p;
		
		type CaseIter<T> = NestedPredBranchIter<Self, T>;
		
		fn comb_split<K: CombKind>(
			(a, b): &'static SystemParamItem<<Self::AllParams as PredCombinator>::Param>,
			Nested(a_input, b_input): Self::Input,
			kind: K,
		) -> Self::CombSplit<K> {
			(
				A::comb(a, kind.pal(), a_input),
				[
					B::comb_split(b, b_input.clone(), CombBranch::A(kind.pal())),
					B::comb_split(b, b_input, CombBranch::B(kind)),
				]
			)
		}
		
		fn case_iter<T>((id, node): Self::Case<T>) -> Self::CaseIter<T> {
			let mut iter = node.into_iter();
			let sub_iter = iter.next().map(B::case_iter);
			NestedPredBranchIter { id, iter, sub_iter }
		}
		
		fn new_comb<'p, T, K>(state: PredSubState2<'p, T, Self, K>) -> Self::Comb<'p, T, K>
		where
			T: Prediction + 'p,
			K: CombKind,
			Self::Branch: 'p,
		{
			NestedPredComb2::new(state)
		}
	}
	
	impl<A, const N: usize, B> PredBranch for NestedPerm<PredArrayComb<A, N>, B>
	where
		A: PredCombinator,
		A::Id: Ord,
		B: PredPermBranch<Output = A>,
	{
		type Param = PredArrayComb<A, N>;
		type Branch = B;
		type Case<T> = (<PredArrayComb<A, N> as PredCombinator>::Id, Node<B::Case<T>>);
		type AllParams = PredPairComb<PredArrayComb<A, N>, B::AllParams>;
		type Id = NestedPerm<<PredArrayComb<A, N> as PredCombinator>::Id, B::Id>;
		type Input = NestedPerm<<PredArrayComb<A, N> as PredCombinator>::Input, B::Input>;
		type CombSplit<K: CombKind> = (
			<Self::Param as PredCombinator>::Comb<K>,
			[B::CombSplit<CombBranch<K::Pal, K>>; 2],
		);
		
		type Item<'p, T, K> = PredSubState2<'p, T, B, CombBranch<K::Pal, K>>
		where
			T: Prediction + 'p,
			K: CombKind,
			Self::Branch: 'p;
		
		type Comb<'p, T, K> = NestedPermPredComb2<'p, T, PredArrayComb<A, N>, B, K>
		where
			T: Prediction + 'p,
			K: CombKind,
			Self::Branch: 'p;
		
		type CaseIter<T> = NestedPredBranchIter<Self, T>;
		
		fn comb_split<K: CombKind>(
			(a, b): &'static SystemParamItem<<Self::AllParams as PredCombinator>::Param>,
			NestedPerm(a_input, b_input): Self::Input,
			kind: K,
		) -> Self::CombSplit<K> {
			let mut comb = PredArrayComb::<A, N>::comb(a, kind, a_input);
			comb.is_nested = true;
			(
				comb,
				[
					B::comb_split(b, b_input.clone(), CombBranch::A(kind.pal())),
					B::comb_split(b, b_input, CombBranch::B(kind)),
				]
			)
		}
		
		fn case_iter<T>((id, node): Self::Case<T>) -> Self::CaseIter<T> {
			let mut iter = node.into_iter();
			let sub_iter = iter.next().map(B::case_iter);
			NestedPredBranchIter { id, iter, sub_iter }
		}
		
		fn new_comb<'p, T, K>(state: PredSubState2<'p, T, Self, K>) -> Self::Comb<'p, T, K>
		where
			T: Prediction + 'p,
			K: CombKind,
			Self::Branch: 'p,
		{
			NestedPermPredComb2::new(state)
		}
	}
}

/// ...
pub trait PredNodeCase {
	type Id;
}

impl<I, T> PredNodeCase for PredStateCase<I, T> {
	type Id = I;
}

impl<I, T> PredNodeCase for (I, Node<T>)
where
	I: PredId,
	T: PredNodeCase,
{
	type Id = I;
}

/// ...
pub struct NestedPredBranchIter<P: PredBranch, T> {
	id: <P::Param as PredCombinator>::Id,
	iter: NodeIter<<P::Branch as PredBranch>::Case<T>>,
	sub_iter: Option<<P::Branch as PredBranch>::CaseIter<T>>,
}

impl<A, B, T> Iterator for NestedPredBranchIter<Nested<A, B>, T>
where
	A: PredCombinator,
	B: PredBranch,
{
	type Item = PredStateCase<Nested<A::Id, B::Id>, T>;
	fn next(&mut self) -> Option<Self::Item> {
		while let Some(iter) = self.sub_iter.as_mut() {
			if let Some(case) = iter.next() {
				let (id, pred) = case.into_parts();
				return Some(PredStateCase {
					id: Nested(self.id, id),
					pred,
				})
			}
			self.sub_iter = self.iter.next()
				.map(B::case_iter);
		}
		None
	}
	// fn size_hint(&self) -> (usize, Option<usize>) {
	// 	todo!()
	// }
}

impl<A, const N: usize, B, T> Iterator
	for NestedPredBranchIter<NestedPerm<PredArrayComb<A, N>, B>, T>
where
	A: PredCombinator,
	A::Id: Ord,
	B: PredPermBranch<Output = A>,
{
	type Item = PredStateCase<NestedPerm<<PredArrayComb<A, N> as PredCombinator>::Id, B::Id>, T>;
	fn next(&mut self) -> Option<Self::Item> {
		while let Some(iter) = self.sub_iter.as_mut() {
			if let Some(case) = iter.next() {
				let (id, pred) = case.into_parts();
				let id = NestedPerm(self.id, id);
				// !!! id.sort();
				return Some(PredStateCase { id, pred })
			}
			self.sub_iter = self.iter.next()
				.map(B::case_iter);
		}
		None
	}
	// fn size_hint(&self) -> (usize, Option<usize>) {
	// 	todo!()
	// }
}

/// ...
pub struct PredNode2<P: PredBranch, T> {
	pub(crate) inner: Node<P::Case<T>>,
}

impl<P: PredBranch, T> IntoIterator for PredNode2<P, T> {
	type Item = PredStateCase<P::Id, T>;
	type IntoIter = PredNodeIter2<P, T>;
	fn into_iter(self) -> Self::IntoIter {
		let mut iter = self.inner.into_iter();
		let case_iter = iter.next().map(P::case_iter);
		PredNodeIter2 { iter, case_iter }
	}
}

/// ...
pub struct PredNodeIter2<P: PredBranch, T> {
	iter: NodeIter<P::Case<T>>,
	case_iter: Option<P::CaseIter<T>>,
}

impl<P: PredBranch, T> Iterator for PredNodeIter2<P, T> {
	type Item = PredStateCase<P::Id, T>;
	fn next(&mut self) -> Option<Self::Item> {
		while let Some(case_iter) = self.case_iter.as_mut() {
			if let Some(case) = case_iter.next() {
				return Some(case)
			}
			self.case_iter = self.iter.next()
				.map(P::case_iter);
		}
		None
	}
	// fn size_hint(&self) -> (usize, Option<usize>) {
	// 	todo!()
	// }
}

/// Types that can be used to query for a specific entity.
pub trait PredFetchData {
	type Id: PredId;
	type Output<'w>;
	unsafe fn get_inner(world: UnsafeWorldCell, id: Self::Id) -> Self::Output<'_>;
	// !!! Could take a dynamically-dispatched ID and attempt downcasting
	// manually. Return an `Option<Self::Output>` for whether it worked. This
	// would allow for query types that accept multiple IDs; support `() -> ()`.
}

mod _pred_fetch_data_impls {
	use super::*;
	
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
	
	impl<A: PredFetchData> PredFetchData for (A,) {
		type Id = (A::Id,);
		type Output<'w> = (A::Output<'w>,);
		unsafe fn get_inner(world: UnsafeWorldCell, (a,): Self::Id) -> Self::Output<'_> {
			(A::get_inner(world, a),)
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
	
	impl<A, B> PredFetchData for Nested<A, B>
	where
		A: PredFetchData,
		B: PredFetchData,
	{
		type Id = Nested<A::Id, B::Id>;
		type Output<'w> = Nested<A::Output<'w>, B::Output<'w>>;
		unsafe fn get_inner(world: UnsafeWorldCell, Nested(a, b): Self::Id) -> Self::Output<'_> {
			Nested(A::get_inner(world, a), B::get_inner(world, b))
		}
	}
	
	impl<A, B> PredFetchData for NestedPerm<A, B>
	where
		A: PredFetchData,
		B: PredFetchData,
	{
		type Id = NestedPerm<A::Id, B::Id>;
		type Output<'w> = NestedPerm<A::Output<'w>, B::Output<'w>>;
		unsafe fn get_inner(world: UnsafeWorldCell, NestedPerm(a, b): Self::Id) -> Self::Output<'_> {
			NestedPerm(A::get_inner(world, a), B::get_inner(world, b))
		}
	}
}

/// Prediction data fed as a parameter to an event's systems.
pub struct PredFetch<'world, D: PredFetchData> {
    inner: D::Output<'world>,
	time: Duration,
}

mod _pred_fetch_impls {
	use super::*;
	
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
						"!!! parameter is for wrong ID type. got {:?} in {:?}",
						std::any::type_name::<D::Id>(),
						system_meta.name(),
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
}

/// ...
/// 
/// ## Types of Input
/// 
/// Input ([`In`])
/// ```text
/// In<T> -> T
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
/// [`Default`] ([`RangeFull`], [`RangeTo`], [`RangeFrom`])
/// ```text
/// .. -> T where T: Default
/// ..(A,) -> (*, B) where A: IntoInput<B>, *: Default
/// (A,).. -> (B, *) where A: IntoInput<B>, *: Default
/// ```
/// 
/// Single vs Nested
/// 
/// ...
pub trait IntoInput<I> {
	fn into_input(self) -> I;
}

mod _into_input_impls {
	use super::*;
	
	impl<T> IntoInput<T> for In<T> {
		fn into_input(self) -> T {
			self.0
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
	
	impl<A, B> IntoInput<Single<B>> for Single<A>
	where
		A: IntoInput<B>,
	{
		fn into_input(self) -> Single<B> {
			let Single(a) = self;
			Single(a.into_input())
		}
	}
	
	impl<A0, B0, A1, B1> IntoInput<Nested<B0, B1>> for Nested<A0, A1>
	where
		A0: IntoInput<B0>,
		A1: IntoInput<B1>,
	{
		fn into_input(self) -> Nested<B0, B1> {
			let Nested(a, b) = self;
			Nested(a.into_input(), b.into_input())
		}
	}
	
	impl<A0, B0, A1, B1> IntoInput<NestedPerm<B0, B1>> for NestedPerm<A0, A1>
	where
		A0: IntoInput<B0>,
		A1: IntoInput<B1>,
	{
		fn into_input(self) -> NestedPerm<B0, B1> {
			let NestedPerm(a, b) = self;
			NestedPerm(a.into_input(), b.into_input())
		}
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
		for<'w> <PredPairComb<QueryComb<'w, &'static Test>, QueryComb<'w, &'static TestB>> as PredCombinator>::Input:
			Default
			+ IntoInput<<PredPairComb<QueryComb<'w, &'static Test>, QueryComb<'w, &'static TestB>> as PredCombinator>::Input>
			+ Send + Sync + 'static,
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
			state: PredState2<DynPred, Single<PredPairComb<QueryComb<&'static Test>, QueryComb<&'static TestB>>>>,
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
							if let Some((_, (x, y))) = iter.next() {
								assert_eq!((*x, *y), (a, b));
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
						assert_eq!(iter.size_hint(), (B, Some(count)));
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
			state: PredState2<DynPred, Nested<QueryComb<&'static Test>, Single<QueryComb<&'static TestB>>>>,
			a_query: Query<Ref<Test>>,
			b_query: Query<Ref<TestB>>,
			mut index: system::Local<usize>,
		| {
			let mut iter = state.into_iter();
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
							assert_eq!(**x, *a);
							for b in &b_query {
								// This assumes `iter` and `QueryIter` always
								// produce the same order.
								if let Some((_, y)) = iter.next() {
									assert_eq!(**y, *b);
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
						assert_eq!(**a, *a_ref);
						let iter = state.into_iter();
						if DetectChanges::is_changed(&a_ref) {
							assert!(update_vec.contains(&a.0));
							assert_eq!(iter.size_hint(), (B, Some(B)));
							for ((_, b), b_ref) in iter.zip(&b_query) {
								assert_eq!(**b, *b_ref);
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
	
	fn test_array<
		const N: usize,
		const R: usize,
		const S: usize,
	>(update_list: &[usize])
	where
		for<'w> <PredArrayComb<QueryComb<'w, &'static Test>, R> as PredCombinator>::Input:
			Default + IntoInput<<PredArrayComb<QueryComb<'w, &'static Test>, R> as PredCombinator>::Input> + Send + Sync + 'static,
		for<'w> <PredArrayComb<QueryComb<'w, &'static Test>, S> as PredCombinator>::Input:
			Default + IntoInput<<PredArrayComb<QueryComb<'w, &'static Test>, S> as PredCombinator>::Input> + Send + Sync + 'static,
	{
		use crate::*;
		
		assert_eq!(R, S+1);
		
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
			state: PredState2<DynPred, Single<PredArrayComb<QueryComb<&'static Test>, R>>>,
			query: Query<&Test>,
			mut index: system::Local<usize>,
		| {
			let mut iter = state.into_iter();
			match *index {
				0 => { // Full
					assert_eq!(iter.size_hint(), (n_choose_r, Some(n_choose_r)));
					let mut n = 0;
					for ((_, a), mut b) in iter
						.zip(query.iter_combinations::<R>())
					{
						// This assumes `iter` and `Query::iter_combinations`
						// will always return in the same order.
						let mut a = a.into_iter()
							.map(Fetch::into_inner)
							.collect::<Vec<_>>();
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
						let a = a.into_iter()
							.map(Fetch::into_inner)
							.collect::<Vec<_>>();
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
		type Param<'w, const S: usize> = NestedPerm<PredArrayComb<QueryComb<'w, &'static Test>, 1>, Single<PredArrayComb<QueryComb<'w, &'static Test>, S>>>;
		app.add_chime_events((move |
			state: PredState2<DynPred, Param<'_, S>>,
			query: Query<Ref<Test>>,
			mut index: system::Local<usize>,
		| {
			let mut iter = state.into_iter();
			match *index {
				0 => { // Full
					let count = N.checked_sub(R).map(|x| x + 1).unwrap_or(0);
					assert_eq!(iter.size_hint(), (count, Some(count)));
					let mut n = 0;
					for ((state, [a]), b) in iter.zip(&query) {
						// This assumes `iter` and `Query` will always return
						// in the same order.
						assert_eq!(**a, *b);
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
					assert_eq!(iter.size_hint(), (
						(update_vec.len() + 1).saturating_sub(R),
						Some(count)
					));
					let mut n = 0;
					for ((state, [a]), b) in iter.zip(&query) {
						// This assumes `iter` and `Query` will always return
						// in the same order.
						assert_eq!(**a, *b);
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
		test_array::<10, 4, 3>(&[0, 4, 6]);
		test_array::<200, 2, 1>(&[10, 17, 100, 101, 102, 103, 104, 105, 199]);
		
		 // Weird Cases:
		test_array::<10, 10, 9>(&[]);
		// test_array::<16, 1, 0>(&[]);
		test_array::<0, 2, 1>(&[]);
		// test_array::<10, 0, 0>(&[]);
		// test_array::<0, 0, 0>(&[]);
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