use std::hash::Hash;
use std::ops::{Deref, RangeTo, RangeFrom, DerefMut};
use bevy_ecs::change_detection::{DetectChanges, Ref, Res};
use bevy_ecs::component::Component;
use bevy_ecs::prelude::Resource;
use bevy_ecs::query::{ArchetypeFilter, QueryData, ReadOnlyQueryData, WorldQuery};
use bevy_ecs::system::{SystemParam, SystemParamItem};
use chime::pred::Prediction;
use crate::node::*;
use crate::comb::*;

/// "Etcetera" aka "And the rest". Used for [`IntoInput`] and [`FetchData`].
pub type Etc = std::ops::RangeFull;

/// A hashable unique identifier for a case of prediction.
pub trait PredId:
	Copy + Clone + Eq + Hash + Send + Sync + 'static
	+ std::fmt::Debug // ??? Tentative
{}

impl<T> PredId for T
where
	T: Copy + Clone + Eq + Hash + Send + Sync + 'static
	+ std::fmt::Debug
{}

/// Maps the [`QueryData`] parameter of [`Each`] to a [`PredItemRef`] type.
/// Otherwise equivalent to [`ReadOnlyQueryData`].
pub trait EachData: ReadOnlyQueryData {
	type ItemRef: ReadOnlyQueryData + 'static;
}

impl<'a, T> EachData for &'a T
where
	T: Component,
{
	type ItemRef = Ref<'static, T>;
}

impl<A, B> EachData for (A, B)
where
	A: EachData,
	B: EachData,
{
	type ItemRef = (A::ItemRef, B::ItemRef);
}

/// Single-item equivalent of [`bevy_ecs::system::Query`]. A [`PredItem`].
#[derive(Debug)]
pub struct Each<'w, D: QueryData, F = ()> {
	inner: D::Item<'w>,
	_filter: std::marker::PhantomData<F>,
}

impl<'w, D: QueryData, F> Each<'w, D, F> {
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

impl<'w, D: QueryData, F> Deref for Each<'w, D, F> {
	type Target = D::Item<'w>;
	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

impl<'w, D: QueryData, F> DerefMut for Each<'w, D, F> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.inner
	}
}

/// A [`PredId`], used as a [`PredItem`] or [`FetchData`].
#[derive(Copy, Clone)]
pub struct EachIn<I>(pub I);

/// Modifier for a combinator to avoid repeated combinations ([A,B] == [B,A]).
#[derive(Copy, Clone)]
pub struct Unique<T>(pub T);

/// Parameter of a [`PredCaseFn`]. An item fetched from the
/// [`bevy_ecs::system::World`] for making predictions against other items.
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
	
	impl<'w, D, F> PredItem for Each<'w, D, F>
	where
		D: QueryData,
		D::Item<'w>: PredItem,
	{
		fn clone(&self) -> Self {
			Each::new(self.inner.clone())
		}
	}
	
	impl<'w, 's, D, F> PredItem for bevy_ecs::system::Query<'w, 's, D, F>
	where
		D: bevy_ecs::query::ReadOnlyQueryData,
		F: bevy_ecs::query::QueryFilter,
	{
		fn clone(&self) -> Self {
			unsafe {
				// SAFETY: This is probably safe, assuming that the Query is
				// read-only. I'm not 100% sure but it works for now. I'd prefer
				// a safe clone; maybe I can make this an `&Query` impl later.
				std::ptr::read(self)
			}
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
	
	impl<T> PredItem for Misc<T>
	where
		T: PredItem,
	{
		fn clone(&self) -> Self {
			let Misc(inner) = self;
			Misc(inner.clone())
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
	
	impl<T, const N: usize> PredItem for Unique<[T; N]>
	where
		T: PredItem
	{
		fn clone(&self) -> Self {
			let Unique(inner) = self;
			Unique(inner.each_ref()
				.map(T::clone))
		}
	}
	
	impl<I> PredItem for EachIn<I>
	where
		I: PredId
	{
		fn clone(&self) -> Self {
			*self
		}
	}
}

/// Type inference addon to [`PredItem`], for mapping to a [`PredCombinator`].  
pub trait PredItem2<P: PredCombinator<Item_= Self> + ?Sized>: PredItem {}

mod _pred_item2_impls {
	use super::*;
	
	impl PredItem2<EmptyComb> for () {}
	
	impl<'w, D, F> PredItem2<FetchComb<'w, D, F>> for Each<'w, D, F>
	where
		D: EachData,
		F: ArchetypeFilter + 'static,
		D::Item<'w>: PredItem,
		<D::ItemRef as WorldQuery>::Item<'w>: PredItemRef<Item = D::Item<'w>>,
	{}
	
	impl<'w, 's, D, F> PredItem2<SystemParamComb<Self>>
		for bevy_ecs::system::Query<'w, 's, D, F>
	where
		D: EachData + 'static,
		F: ArchetypeFilter + 'static,
		D::Item<'w>: PredItem,
		<D::ItemRef as WorldQuery>::Item<'w>: PredItemRef<Item = D::Item<'w>>,
	{}
	
	impl<'w, T: Resource> PredItem2<SystemParamComb<Res<'w, T>>> for Res<'w, T> {}
	
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
	
	impl<T, P, const N: usize> PredItem2<PredArrayComb<P, N>> for Unique<[T; N]>
	where
		T: PredItem2<P>,
		P: PredCombinator<Item_= T>,
		P::Id: Ord,
	{}
	
	impl<T> PredItem2<PredIdComb<T>> for EachIn<T::Item>
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

/// [`PredItem`] with updated state, used specifically for [`CombIter`].
pub trait PredItemRef {
	type Item: PredItem;
	fn into_item(self) -> Self::Item;
	
	/// Whether this item is in need of a prediction update.
	fn is_updated(item: &mut Self) -> bool;
}

mod _pred_item_ref_impls {
	use super::*;
	
	impl<'w, T: 'static> PredItemRef for Ref<'w, T> {
		type Item = &'w T;
		fn into_item(self) -> Self::Item {
			self.into_inner()
		}
		fn is_updated(item: &mut Self) -> bool {
			DetectChanges::is_changed(item)
		}
	}
	
	impl<'w, 's, D, F> PredItemRef for bevy_ecs::system::Query<'w, 's, D, F>
	where
		D: EachData + 'static,
		F: ArchetypeFilter + 'static,
		D::Item<'w>: PredItem,
		<D::ItemRef as WorldQuery>::Item<'w>: PredItemRef<Item = D::Item<'w>>,
	{
		type Item = Self;
		fn into_item(self) -> Self::Item {
			self
		}
		fn is_updated(item: &mut Self) -> bool {
			let mut query_lens = item.transmute_lens_filtered::<D::ItemRef, F>();
			let query = &query_lens.query();
			let query: &'w bevy_ecs::system::Query<'static, 'static, D::ItemRef, F> = unsafe {
				// SAFETY: As I understand it, the only reason that the lifetime
				// of the `Query` is bound to the `QueryLens` is because mutable
				// queries shouldn't overlap. However, upcasting the lifetime
				// should be fine because `D::ItemRef` is read-only. Also, the
				// upcasted query is consumed entirely inside this function.
				std::mem::transmute(query)
			};
			for mut sub_item in query {
				if PredItemRef::is_updated(&mut sub_item) {
					return true
				}
			}
			false
		}
	}
	
	impl<'w, R: Resource> PredItemRef for Res<'w, R> {
		type Item = Self;
		fn into_item(self) -> Self::Item {
			self
		}
		fn is_updated(item: &mut Self) -> bool {
			DetectChanges::is_changed(item)
		}
	}
	
	impl PredItemRef for () {
		type Item = ();
		fn into_item(self) -> Self::Item {
			self
		}
		fn is_updated(_item: &mut Self) -> bool {
			false
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
		fn is_updated(item: &mut Self) -> bool {
			let (a, b) = item;
			A::is_updated(a) || B::is_updated(b)
		}
	}
}

/// Collects predictions from "when" systems for later compilation.
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
		<Self::Param as PredCombinator>::Item_,
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
		type Input = A::Input;
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
			input: Self::Input,
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

/// ...
pub struct Fetch<T>(pub T);

/// ...
pub trait ChimeSystemParamGroup<I: PredId> {
	type Param: SystemParam;
	type Item<'w, 's>;
	fn fetch_param<'w, 's>(
		param: SystemParamItem<'w, 's, Self::Param>,
		id: I,
		time: std::time::Duration,
	) -> Self::Item<'w, 's>;
}

macro_rules! impl_chime_system_param_group {
    ($($param:ident $(, $rest:ident)*)?) => {
		impl_chime_system_param_group!(@impl $($param $(, $rest)*)?);
		$(impl_chime_system_param_group!($($rest),*);)?
    };
	(@impl $($param:ident),*) => {
		impl<I, $($param,)*> ChimeSystemParamGroup<I> for ($($param,)*)
		where
			I: PredId,
			$($param: ChimeSystemParam<I>,)*
		{
			type Param = ($($param::Param,)*);
			type Item<'w, 's> = ($($param::Item<'w, 's>,)*);
			#[allow(unused_variables)]
			#[allow(clippy::unused_unit)]
			fn fetch_param<'w, 's>(
				($($param,)*): SystemParamItem<'w, 's, Self::Param>,
				id: I,
				time: std::time::Duration,
			) -> Self::Item<'w, 's> {
				($($param::fetch_param($param, id, time),)*)
			}
		}
    };
}

impl_chime_system_param_group!(
	_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15
);

/// ...
pub trait ChimeSystemParam<I: PredId> {
	type Param: SystemParam;
	type Item<'w, 's>;
	fn fetch_param<'w, 's>(
		param: SystemParamItem<'w, 's, Self::Param>,
		id: I,
		time: std::time::Duration,
	) -> Self::Item<'w, 's>;
}

mod _pred_param_impls {
	use super::{ChimeSystemParam, Fetch, FetchData, PredId};
	use bevy_ecs::system::{SystemParam, SystemParamItem};
	
	impl<I, T> ChimeSystemParam<I> for T
	where
		I: PredId,
		T: SystemParam,
	{
		type Param = T;
		type Item<'w, 's> = SystemParamItem<'w, 's, T>;
		fn fetch_param<'w, 's>(
			param: SystemParamItem<'w, 's, Self::Param>,
			_id: I,
			_time: std::time::Duration,
		) -> Self::Item<'w, 's> {
			param
		}
	}
	
	impl<I, T> ChimeSystemParam<I> for Fetch<T>
	where
		I: PredId,
		T: FetchData<I>,
	{
		type Param = T::Param;
		type Item<'w, 's> = Fetch<T::Item<'w, 's>>;
		fn fetch_param<'w, 's>(
			param: SystemParamItem<'w, 's, Self::Param>,
			id: I,
			time: std::time::Duration,
		) -> Self::Item<'w, 's> {
			Fetch(T::fetch_item(param, id, time))
		}
	}
}

/// ...
pub trait FetchData<I: PredId> {
	type Param: SystemParam;
	type Item<'w, 's>;
	fn fetch_item<'w, 's>(
		param: SystemParamItem<'w, 's, Self::Param>,
		id: I,
		time: std::time::Duration,
	) -> Self::Item<'w, 's>;
}

mod _pred_fetch_data_impls {
	use std::time::Duration;
	use super::{Each, EachIn, Etc, Nested, FetchData, PredId};
	use bevy_ecs::entity::Entity;
	use bevy_ecs::query::QueryData;
	use bevy_ecs::system::{Query, Res, ResMut, Resource, SystemParamItem};
	use chime::{Moment, ResMoment, ResMomentMut};
	
	impl<I: PredId> FetchData<I> for () {
		type Param = ();
		type Item<'w, 's> = ();
		fn fetch_item<'w, 's>(
			_param: SystemParamItem<'w, 's, Self::Param>,
			_id: I,
			_time: Duration,
		) -> Self::Item<'w, 's> {}
	}
	
	impl<I: PredId> FetchData<I> for Etc {
		type Param = ();
		type Item<'w, 's> = Etc;
		fn fetch_item<'w, 's>(
			_param: SystemParamItem<'w, 's, Self::Param>,
			_id: I,
			_time: Duration,
		) -> Self::Item<'w, 's> {
			..
		}
	}
	
	impl<T> FetchData<Entity> for Each<'_, T>
	where
		T: QueryData + 'static,
	{
		type Param = Query<'static, 'static, T>;
		type Item<'w, 's> = Each<'w, T>;
		fn fetch_item<'w, 's>(
			mut param: SystemParamItem<'w, 's, Self::Param>,
			id: Entity,
			_time: Duration,
		) -> Self::Item<'w, 's> {
			let m = param.get_mut(id)
				.expect("should exist");
			Each::new(unsafe {
				// SAFETY: Upcasting this lifetime should be fine because the
				// query is being consumed by this `fetch_item`. My
				// understanding is that `Query::get_inner_mut` doesn't exist
				// only because it could create overlapping mutable refs.
				std::mem::transmute(m)
			})
		}
	}
	
	impl<T: Resource> FetchData<()> for Res<'_, T> {
		type Param = Res<'static, T>;
		type Item<'w, 's> = Res<'w, T>;
		fn fetch_item<'w, 's>(
			param: SystemParamItem<'w, 's, Self::Param>,
			_id: (),
			_time: Duration,
		) -> Self::Item<'w, 's> {
			param
		}
	}
	
	impl<T: Resource> FetchData<()> for ResMut<'_, T> {
		type Param = ResMut<'static, T>;
		type Item<'w, 's> = ResMut<'w, T>;
		fn fetch_item<'w, 's>(
			param: SystemParamItem<'w, 's, Self::Param>,
			_id: (),
			_time: Duration,
		) -> Self::Item<'w, 's> {
			param
		}
	}
	
	impl<T, const N: usize> FetchData<[Entity; N]> for [Each<'_, T>; N]
	where
		T: QueryData + 'static,
	{
		type Param = Query<'static, 'static, T>;
		type Item<'w, 's> = [Each<'w, T>; N];
		fn fetch_item<'w, 's>(
			mut param: SystemParamItem<'w, 's, Self::Param>,
			id: [Entity; N],
			_time: Duration,
		) -> Self::Item<'w, 's> {
			let m = param.get_many_mut(id)
				.expect("should exist");
			let m: [T::Item<'w>; N] = unsafe {
				std::mem::transmute(m)
			};
			m.map(Each::new)
		}
	}
	
	impl<I, A,> FetchData<(I,)> for (A,)
	where
		I: PredId,
		A: FetchData<I>,
	{
		type Param = (A::Param,);
		type Item<'w, 's> = (A::Item<'w, 's>,);
		fn fetch_item<'w, 's>(
			(a,): SystemParamItem<'w, 's, Self::Param>,
			(i,): (I,),
			time: Duration,
		) -> Self::Item<'w, 's> {
			(A::fetch_item(a, i, time),)
		}
	}
	
	impl<I, J, A, B,> FetchData<(I, J,)> for (A, B,)
	where
		I: PredId,
		J: PredId,
		A: FetchData<I>,
		B: FetchData<J>,
	{
		type Param = (A::Param, B::Param,);
		type Item<'w, 's> = (A::Item<'w, 's>, B::Item<'w, 's>,);
		fn fetch_item<'w, 's>(
			(a, b,): SystemParamItem<'w, 's, Self::Param>,
			(i, j,): (I, J,),
			time: Duration,
		) -> Self::Item<'w, 's> {
			(A::fetch_item(a, i, time), B::fetch_item(b, j, time),)
		}
	}
	
	impl<I: PredId> FetchData<I> for EachIn<I> {
		type Param = ();
		type Item<'w, 's> = Self;
		fn fetch_item<'w, 's>(
			_param: SystemParamItem<'w, 's, Self::Param>,
			id: I,
			_time: Duration,
		) -> Self::Item<'w, 's> {
			EachIn(id)
		}
	}
	
	impl<T> FetchData<()> for ResMoment<'_, T>
	where
		T: Moment,
		T::Flux: Resource + Clone,
	{
		type Param = ResMoment<'static, T>;
		type Item<'w, 's> = ResMoment<'w, T>;
		fn fetch_item<'w, 's>(
			param: SystemParamItem<'w, 's, Self::Param>,
			_id: (),
			_time: Duration,
		) -> Self::Item<'w, 's> {
			param
		}
	}
	
	impl<T> FetchData<()> for ResMomentMut<'_, T>
	where
		T: Moment,
		T::Flux: Resource + Clone,
	{
		type Param = ResMomentMut<'static, T>;
		type Item<'w, 's> = ResMomentMut<'w, T>;
		fn fetch_item<'w, 's>(
			param: SystemParamItem<'w, 's, Self::Param>,
			_id: (),
			_time: Duration,
		) -> Self::Item<'w, 's> {
			param
		}
	}
	
	impl<I, J, A, B> FetchData<Nested<I, J>> for Nested<A, B>
	where
		I: PredId,
		J: PredId,
		A: FetchData<I>,
		B: FetchData<J>,
	{
		type Param = (A::Param, B::Param);
		type Item<'w, 's> = Nested<A::Item<'w, 's>, B::Item<'w, 's>>;
		fn fetch_item<'w, 's>(
			(a, b): SystemParamItem<'w, 's, Self::Param>,
			Nested(i, j): Nested<I, J>,
			time: Duration,
		) -> Self::Item<'w, 's> {
			Nested(A::fetch_item(a, i, time), B::fetch_item(b, j, time))
		}
	}
}

/// For input to [`crate::AddChimeEvents::add_chime_events`].
#[derive(Copy, Clone, Default)]
pub struct In<T>(pub T);

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
/// [`Default`] ([`Etc`], [`RangeTo`], [`RangeFrom`])
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
	
	impl<T> IntoInput<T> for Etc
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
	use bevy::prelude::Query;
	
	#[derive(Component, Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
	struct Test(usize);
	
	#[derive(Component, Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
	struct TestB(usize);
	
	fn test_pair<const A: usize, const B: usize>(
		update_list: &[usize],
		b_update_list: &[usize],
	)
	where
		for<'w> <PredPairComb<FetchComb<'w, &'static Test>, FetchComb<'w, &'static TestB>> as PredCombinator>::Input:
			Default
			+ IntoInput<<PredPairComb<FetchComb<'w, &'static Test>, FetchComb<'w, &'static TestB>> as PredCombinator>::Input>
			+ Send + Sync + 'static,
	{
		use crate::*;
		
		let mut app = App::new();
		app.insert_resource::<Time>(Time::default());
		app.add_plugins(ChimePlugin);
		
		for i in 0..A {
			app.world_mut().spawn(Test(i));
		}
		for i in 0..B {
			app.world_mut().spawn(TestB(i));
		}
		
		 // Setup [`PredPairComb`] Testing:
		let update_vec = update_list.to_vec();
		let b_update_vec = b_update_list.to_vec();
		struct TestRunA<const A: usize, const B: usize>([Vec<usize>; 2]);
		impl<const A: usize, const B: usize> PredSystem<
			DynPred,
			Single<PredPairComb<FetchComb<'_, &'static Test>, FetchComb<'_, &'static TestB>>>,
			(
				Query<'_, '_, &'static Test>,
				Query<'_, '_, &'static TestB>,
				system::Local<'_, usize>
			)
		> for TestRunA<A, B> {
			fn run<K: CombKind>(
				&self,
				state: PredSubState2<DynPred, Single<PredPairComb<FetchComb<&'static Test>, FetchComb<&'static TestB>>>, K>,
				(a_query, b_query, mut index): (
					Query<&Test>,
					Query<&TestB>,
					system::Local<usize>,
				),
			) {
				let TestRunA([update_vec, b_update_vec]) = self;
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
			}
		}
		app.add_chime_events(ChimeEventBuilder::new(TestRunA::<A, B>([update_vec, b_update_vec])).on_begin(|| {}));
		
		
		 // Setup [`PredSubState::outer_iter`] Testing:
		let update_vec = update_list.to_vec();
		let b_update_vec = b_update_list.to_vec();
		struct TestRunB<const A: usize, const B: usize>([Vec<usize>; 2]);
		impl<const A: usize, const B: usize> PredSystem<
			DynPred,
			Nested<FetchComb<'_, &'static Test>, Single<FetchComb<'_, &'static TestB>>>,
			(
				Query<'_, '_, Ref<'static, Test>>,
				Query<'_, '_, Ref<'static, TestB>>,
				system::Local<'_, usize>
			)
		> for TestRunB<A, B> {
			fn run<K: CombKind>(
				&self,
				state: PredSubState2<DynPred, Nested<FetchComb<&'static Test>, Single<FetchComb<&'static TestB>>>, K>,
				(a_query, b_query, mut index): (
					Query<Ref<Test>>,
					Query<Ref<TestB>>,
					system::Local<usize>,
				),
			) {
				let TestRunB([update_vec, b_update_vec]) = self;
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
			}
		}
		app.add_chime_events(ChimeEventBuilder::new(TestRunB::<A, B>([update_vec, b_update_vec])).on_begin(|| {}));
		
		 // Run Tests:
		app.world_mut().run_schedule(ChimeSchedule);
		app.world_mut().run_schedule(ChimeSchedule);
		for mut test in app.world_mut().query::<&mut Test>()
			.iter_mut(app.world_mut())
		{
			if update_list.contains(&test.0) {
				test.0 = std::hint::black_box(test.0);
			}
		}
		for mut test in app.world_mut().query::<&mut TestB>()
			.iter_mut(app.world_mut())
		{
			if b_update_list.contains(&test.0) {
				test.0 = std::hint::black_box(test.0);
			}
		}
		app.world_mut().run_schedule(ChimeSchedule);
	}
	
	fn test_array<
		const N: usize,
		const R: usize,
		const S: usize,
	>(update_list: &[usize])
	where
		for<'w> <PredArrayComb<FetchComb<'w, &'static Test>, R> as PredCombinator>::Input:
			Default + IntoInput<<PredArrayComb<FetchComb<'w, &'static Test>, R> as PredCombinator>::Input> + Send + Sync + 'static,
		for<'w> <PredArrayComb<FetchComb<'w, &'static Test>, S> as PredCombinator>::Input:
			Default + IntoInput<<PredArrayComb<FetchComb<'w, &'static Test>, S> as PredCombinator>::Input> + Send + Sync + 'static,
	{
		use crate::*;
		
		assert_eq!(R, S+1);
		
		let mut app = App::new();
		app.insert_resource::<Time>(Time::default());
		app.add_plugins(ChimePlugin);
		
		for i in 0..N {
			app.world_mut().spawn(Test(i));
		}
		
		fn choose(n: usize, r: usize) -> usize {
			if n < r || r == 0 {
				return 0
			}
			((1+n-r)..=n).product::<usize>() / (1..=r).product::<usize>()
		}
		
		 // Setup [`PredArrayComb`] Testing:
		let update_vec = update_list.to_vec();
		struct TestRunA<const N: usize, const R: usize, const S: usize>(Vec<usize>);
		impl<const N: usize, const R: usize, const S: usize> PredSystem<
			DynPred,
			Single<PredArrayComb<FetchComb<'_, &'static Test>, R>>,
			(
				Query<'_, '_, &'static Test>,
				system::Local<'_, usize>,
			),
		> for TestRunA<N, R, S> {
			fn run<K: CombKind>(
				&self,
				state: PredSubState2<DynPred, Single<PredArrayComb<FetchComb<&'static Test>, R>>, K>,
				(query, mut index): (
					Query<&Test>,
					system::Local<usize>,
				),
			) {
				let TestRunA(update_vec) = self;
				let n_choose_r = choose(N, R);
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
								.map(Each::into_inner)
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
								.map(Each::into_inner)
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
			}
		}
		app.add_chime_events(ChimeEventBuilder::new(TestRunA::<N, R, S>(update_vec)).on_begin(|| {}));
		
		 // Setup [`PredSubState::outer_iter`] Testing:
		let update_vec = update_list.to_vec();
		struct TestRunB<const N: usize, const R: usize, const S: usize>(Vec<usize>);
		type Param<'w, const S: usize> = NestedPerm<PredArrayComb<FetchComb<'w, &'static Test>, 1>, Single<PredArrayComb<FetchComb<'w, &'static Test>, S>>>;
		impl<const N: usize, const R: usize, const S: usize> PredSystem<
			DynPred,
			Param<'_, S>,
			(
				Query<'_, '_, Ref<'static, Test>>,
				system::Local<'_, usize>,
			)
		> for TestRunB<N, R, S> {
			fn run<K: CombKind>(
				&self, 
				state: PredSubState2<DynPred, Param<'_, S>, K>,
				(query, mut index): (
					Query<Ref<Test>>,
					system::Local<usize>,
				),
			) {
				let TestRunB(update_vec) = self;
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
			}
		}
		app.add_chime_events(ChimeEventBuilder::new(TestRunB::<N, R, S>(update_vec)).on_begin(|| {}));
		
		 // Run Tests:
		app.world_mut().run_schedule(ChimeSchedule);
		app.world_mut().run_schedule(ChimeSchedule);
		for mut test in app.world_mut().query::<&mut Test>()
			.iter_mut(app.world_mut())
		{
			if update_list.contains(&test.0) {
				test.0 = std::hint::black_box(test.0);
			}
		}
		app.world_mut().run_schedule(ChimeSchedule);
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