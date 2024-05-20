use std::rc::Rc;
use bevy_ecs::change_detection::Res;
use bevy_ecs::entity::Entity;
use bevy_ecs::query::{ArchetypeFilter, QueryIter, WorldQuery};
use bevy_ecs::system::{Query, Resource};
use crate::node::{Node, NodeWriter};
use crate::pred::*;

/// Unit types that filter what `PredParam::comb` iterates over.
/// 
/// ```text
///           | ::Pal    | ::Inv    | ::Inv::Pal::Inv
///  None     | None     | All      | None
///  All      | All      | None     | All
///  AllTrue  | AllTrue  | AnyFalse | None
///  AllFalse | AllFalse | AnyTrue  | None
///  AnyTrue  | All      | AllFalse | AnyTrue
///  AnyFalse | All      | AllTrue  | AnyFalse
/// ```
/// 
/// - `None`: No combinations.
/// - `All`: All possible combinations.
/// - `AllTrue`: Combinations where all are true.
/// - `AllFalse`: Combinations where all are false.
/// - `AnyTrue`: Combinations where at least one is true.
/// - `AnyFalse`: Combinations where at least one is false.
/// 
/// ??? Potentially rewrite to define separate associated types for `True` and
/// `False`. Could support kinds like `AllSame` + `AnyDiff` if ever needed.
pub trait CombKind: Copy + 'static {
	/// This is like a helper kind. I'm not really sure how to describe it. It
	/// generally defines a combination superset. A coincidental utility? 
	type Pal: CombKind;
	
	/// The inverse of this kind, filter-wise. Not overlapping; complementary.
	type Inv: CombKind;
	
	/// Filters some state.
	fn has_state(self, state: bool) -> bool;
	
	/// Conversion from `Self` to [`Self::Pal`].
	fn pal(self) -> Self::Pal;
	
	/// Conversion from `Self` to [`Self::Inv`].
	fn inv(self) -> Self::Inv;
	
	/// If [`Self::has_state`] always filters to `true`.
	#[inline]
	fn has_all(self) -> bool {
		self.has_state(true) && self.has_state(false)
	}
	
	/// If [`Self::has_state`] always filters to `false`.
	#[inline]
	fn has_none(self) -> bool {
		!(self.has_state(true) || self.has_state(false))
	}
	
	/// All outputs of [`Self::has_state`].
	#[inline]
	fn states(self) -> [bool; 2] {
		[self.has_state(true), self.has_state(false)]
	}
}

macro_rules! def_comb_kind {
	($name:ident, $pal:ident, $inv:ident, $state:pat => $has_state:expr) => {
		/// See [`CombKind`].
		#[derive(Copy, Clone)]
		pub struct $name;
		
		impl CombKind for $name {
			type Pal = $pal;
			type Inv = $inv;
			
			#[inline]
			fn has_state(self, $state: bool) -> bool {
				$has_state
			}
			
			#[inline]
			fn pal(self) -> Self::Pal {
				$pal
			}
			
			#[inline]
			fn inv(self) -> Self::Inv {
				$inv
			}
		}
	};
}

            // Name,         Pal,          Inv,          Filter
def_comb_kind!(CombNone,     CombNone,     CombAll,      _ => false);
def_comb_kind!(CombAll,      CombAll,      CombNone,     _ => true);
def_comb_kind!(CombAllFalse, CombAllFalse, CombAnyTrue,  x => !x);
def_comb_kind!(CombAnyTrue,  CombAll,      CombAllFalse, x => x);

/// A branching [`CombKind`] for combinators that split into two paths.
#[derive(Copy, Clone)]
pub enum CombBranch<A, B> {
	A(A),
	B(B),
}

impl<A, B> CombKind for CombBranch<A, B>
where
	A: CombKind,
	B: CombKind,
{
	type Pal = CombBranch<A::Pal, B::Pal>;
	type Inv = CombBranch<A::Inv, B::Inv>;
	
	#[inline]
	fn has_state(self, state: bool) -> bool {
		match self {
			Self::A(a) => a.has_state(state),
			Self::B(b) => b.has_state(state),
		}
	}
	
	#[inline]
	fn pal(self) -> Self::Pal {
		match self {
			Self::A(a) => CombBranch::A(a.pal()),
			Self::B(b) => CombBranch::B(b.pal()),
		}
	}
	
	#[inline]
	fn inv(self) -> Self::Inv {
		match self {
			Self::A(a) => CombBranch::A(a.inv()),
			Self::B(b) => CombBranch::B(b.inv()),
		}
	}
}

/// Combinator type produced by `PredParam::comb`.
pub trait PredCombinator<K: CombKind = CombNone>:
	Clone + IntoIterator<Item=Self::Case>
{
	type Id: PredId;
	type Case: PredCombinatorCase<Self::Param, Id=Self::Id>;
	type Param: PredParam<Id = Self::Id>;
	
	fn outer_skip(&mut self, _n: [usize; 2]) {
		// !!! This kinda sucks.
	}
}

impl<K: CombKind> PredCombinator<K> for EmptyComb<K> {
	type Id = ();
	type Case = PredCombCase<(), ()>;
	type Param = EmptyComb;
}

impl<'w, R, K> PredCombinator<K> for ResComb<'w, R, K>
where
	K: CombKind,
	R: Resource,
{
	type Id = ();
	type Case = PredCombCase<Res<'w, R>, ()>;
	type Param = ResComb<'static, R>;
}

impl<'w, T, F, K> PredCombinator<K> for QueryComb<'w, T, F, K>
where
	K: CombKind,
	T: PredParamQueryData + 'static,
	F: ArchetypeFilter + 'static,
	for<'a> <T::ItemRef as WorldQuery>::Item<'a>: PredItemRef2,
{
	type Id = Entity;
	type Case = PredCombCase<
		Fetch<<<T::ItemRef as WorldQuery>::Item<'w> as PredItemRef>::Item, F>,
		Entity,
	>;
	type Param = QueryComb<'static, T, F>;
}

impl<'w, A, K> PredCombinator<K> for PredSingleComb<'w, A, K>
where
	K: CombKind,
	A: PredParam,
{
	type Id = (A::Id,);
	type Case = (A::Case<'w>,);
	type Param = PredSingleComb<'static, A>;
}

impl<'w, A, B, K> PredCombinator<K> for PredPairComb<'w, A, B, K>
where
	K: CombKind,
	A: PredParam,
	B: PredParam,
{
	type Id = (A::Id, B::Id);
	type Case = (A::Case<'w>, B::Case<'w>);
	type Param = (A, B);
}

impl<'w, C, const N: usize, K> PredCombinator<K> for PredArrayComb<'w, C, N, K>
where
	K: CombKind,
	C: PredParam,
	C::Id: Ord,
{
	type Id = [C::Id; N];
	type Case = [C::Case<'w>; N];
	type Param = [C; N];
	
	fn outer_skip(&mut self, index: [usize; 2]) {
		self.a_index += index[0];
		self.b_index += index[1];
	}
}

impl<I, K> PredCombinator<K> for PredIdComb<I>
where
	I: IntoIterator + Clone,
	I::Item: PredId,
	K: CombKind,
{
	type Id = I::Item;
	type Case = PredCombCase<WithId<I::Item>, I::Item>;
	type Param = WithId<I>;
}

/// Combinator for `PredParam` `()` implementation.
#[derive(Clone)]
pub struct EmptyComb<K = CombNone> {
	kind: K,
}

impl<K> EmptyComb<K> {
	pub(crate) fn new(kind: K) -> Self {
		Self { kind }
	}
}

impl<K> IntoIterator for EmptyComb<K>
where
	K: CombKind
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = CombIter<std::iter::Once<((), ())>, K>;
	fn into_iter(self) -> Self::IntoIter {
		CombIter::new(std::iter::once(((), ())), self.kind)
	}
}

/// Combinator for `PredParam` `Res` implementation.
pub struct ResComb<'w, T, K = CombNone>
where
	T: Resource,
{
	inner: Res<'w, T>,
	kind: K,
}

impl<'w, T, K> ResComb<'w, T, K>
where
	T: Resource,
{
	pub(crate) fn new(inner: Res<'w, T>, kind: K) -> Self {
		Self {
			inner,
			kind,
		}
	}
}

impl<T, K> Clone for ResComb<'_, T, K>
where
	T: Resource,
	K: Clone,
{
	fn clone(&self) -> Self {
		Self {
			inner: Res::clone(&self.inner),
			kind: self.kind.clone(),
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
		CombIter::new(std::iter::once((self.inner, ())), self.kind)
	}
}

/// Combinator for `PredParam` `Query` implementation.
pub enum QueryComb<'w, T, F = (), K = CombNone>
where
	T: PredParamQueryData,
	F: ArchetypeFilter + 'static,
	<T::ItemRef as WorldQuery>::Item<'w>: PredItemRef,
{
	Normal {
		inner: &'w Query<'w, 'w, (T::ItemRef, Entity), F>,
		kind: K,
	},
	Cached {
		slice: Rc<[PredCombCase<
			Fetch<<<T::ItemRef as WorldQuery>::Item<'w> as PredItemRef>::Item, F>,
			Entity,
		>]>,
		kind: K,
	},
}

impl<'w, T, F, K> QueryComb<'w, T, F, K>
where
	T: PredParamQueryData,
	F: ArchetypeFilter + 'static,
	K: CombKind,
	<T::ItemRef as WorldQuery>::Item<'w>: PredItemRef,
	Fetch<<<T::ItemRef as WorldQuery>::Item<'w> as PredItemRef>::Item, F>: PredItem,
{
	pub(crate) fn new(inner: &'w Query<'w, 'w, (T::ItemRef, Entity), F>, kind: K) -> Self {
		let comb = Self::Normal { inner, kind };
		if kind.has_state(true) == kind.has_state(false) {
			comb
		} else {
			// !!! Test whether this is faster for static/non-updated queries
			// in general. Most of the time it probably isn't.
			Self::Cached {
				slice: comb.into_iter().collect(),
				kind,
			}
		}
	}
}

impl<'w, T, F, K> Clone for QueryComb<'w, T, F, K>
where
	T: PredParamQueryData,
	F: ArchetypeFilter + 'static,
	K: Clone,
	<T::ItemRef as WorldQuery>::Item<'w>: PredItemRef,
{
	fn clone(&self) -> Self {
		match self {
			Self::Normal { inner, kind } => Self::Normal {
				inner: Clone::clone(inner),
				kind: kind.clone(),
			},
			Self::Cached { slice, kind } => Self::Cached {
				slice: Rc::clone(slice),
				kind: kind.clone(),
			},
		}
	}
}

impl<'w, T, F, K> IntoIterator for QueryComb<'w, T, F, K>
where
	K: CombKind,
	T: PredParamQueryData,
	F: ArchetypeFilter + 'static,
	<T::ItemRef as WorldQuery>::Item<'w>: PredItemRef,
	Fetch<<<T::ItemRef as WorldQuery>::Item<'w> as PredItemRef>::Item, F>: PredItem,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = QueryCombIter<'w, T, F, K>;
	fn into_iter(self) -> Self::IntoIter {
		match self {
			Self::Normal { inner, kind } => QueryCombIter::Normal {
				iter: CombIter::new(inner.iter_inner(), kind),
			},
			Self::Cached { slice, .. } => QueryCombIter::Cached {
				slice,
				index: 0,
			},
		}
	}
}

/// ...
pub enum QueryCombIter<'w, T, F, K>
where
	T: PredParamQueryData,
	F: ArchetypeFilter,
	<T::ItemRef as WorldQuery>::Item<'w>: PredItemRef,
{
	Normal { 
		iter: CombIter<QueryIter<'w, 'w, (T::ItemRef, Entity), F>, K>,
	},
	Cached {
		slice: Rc<[PredCombCase<
			Fetch<<<T::ItemRef as WorldQuery>::Item<'w> as PredItemRef>::Item, F>,
			Entity,
		>]>,
		index: usize,
	},
}

impl<'w, T, F, K> Iterator for QueryCombIter<'w, T, F, K>
where
	T: PredParamQueryData,
	F: ArchetypeFilter,
	K: CombKind,
	<T::ItemRef as WorldQuery>::Item<'w>: PredItemRef,
	Fetch<<<T::ItemRef as WorldQuery>::Item<'w> as PredItemRef>::Item, F>: PredItem,
{
	type Item = PredCombCase<
		Fetch<<<T::ItemRef as WorldQuery>::Item<'w> as PredItemRef>::Item, F>,
		Entity,
	>;
	fn next(&mut self) -> Option<Self::Item> {
		match self {
			Self::Normal { iter } => iter.next().map(|x| match x {
				PredCombCase::Diff(item, id) => PredCombCase::Diff(Fetch::new(item), id),
				PredCombCase::Same(item, id) => PredCombCase::Same(Fetch::new(item), id),
			}),
			Self::Cached { slice, index } => {
				if let Some(case) = slice.get(*index) {
					*index += 1;
					Some(case.clone())
				} else {
					None
				}
			},
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		match self {
			Self::Normal { iter } => iter.size_hint(),
			Self::Cached { slice, index } => {
				let len = slice.len() - index;
				(len, Some(len))
			},
		}
	}
	fn nth(&mut self, n: usize) -> Option<Self::Item> {
		match self {
			Self::Normal { iter } => iter.nth(n).map(|x| match x {
				PredCombCase::Diff(item, id) => PredCombCase::Diff(Fetch::new(item), id),
				PredCombCase::Same(item, id) => PredCombCase::Same(Fetch::new(item), id),
			}),
			Self::Cached { index, .. } => {
				*index += n;
				self.next()
			},
		}
	}
	fn count(self) -> usize where Self: Sized {
		match self {
			Self::Normal { iter } => iter.count(),
			Self::Cached { slice, index } => slice.len() - index,
		}
	}
}

/// `Iterator` of `ResComb`'s `IntoIterator` implementation.
pub struct CombIter<T, K> {
	iter: T,
	kind: K,
}

impl<T, K> CombIter<T, K> {
	pub fn new(iter: T, kind: K) -> Self {
		Self { iter, kind }
	}
}

impl<P, I, T, K> Iterator for CombIter<T, K>
where
	P: PredItemRef,
	I: PredId,
	T: Iterator<Item = (P, I)>,
	K: CombKind,
{
	type Item = PredCombCase<P::Item, I>;
	fn next(&mut self) -> Option<Self::Item> {
		if self.kind.has_none() {
			return None
		}
		for (item, id) in self.iter.by_ref() {
			let is_updated = P::is_updated(&item);
			if self.kind.has_state(is_updated) {
				return Some(if is_updated {
					PredCombCase::Diff(item.into_item(), id)
				} else {
					PredCombCase::Same(item.into_item(), id)
				})
			}
		}
		None
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		match self.kind.states() {
			[false, false] => (0, Some(0)),
			[true, true] => self.iter.size_hint(),
			_ => (0, self.iter.size_hint().1)
		}
	}
}

/// Item of a [`PredCombinator`]'s iterator.
pub trait PredCombinatorCase<P: PredParam<Id = Self::Id> + ?Sized>: Clone {
	type Item: PredItem;
	type Id: PredId;
	fn is_diff(&self) -> bool;
	fn into_parts(self) -> (Self::Item, Self::Id);
}

impl PredCombinatorCase<EmptyComb> for () {
	type Item = ();
	type Id = ();
	fn is_diff(&self) -> bool {
		true
	}
	fn into_parts(self) -> (Self::Item, Self::Id) {
		((), ())
	}
}

impl<T, P, I> PredCombinatorCase<T> for PredCombCase<P, I>
where
	T: PredParam<Id = I>,
	P: PredItem,
	I: PredId,
{
	type Item = P;
	type Id = I;
	fn is_diff(&self) -> bool {
		match self {
			PredCombCase::Diff(..) => true,
			PredCombCase::Same(..) => false,
		}
	}
	fn into_parts(self) -> (Self::Item, Self::Id) {
		let (PredCombCase::Diff(item, id) | PredCombCase::Same(item, id)) = self;
		(item, id)
	}
}

impl<T, C, const N: usize> PredCombinatorCase<[T; N]> for [C; N]
where
	T: PredParam<Id = C::Id>,
	C: PredCombinatorCase<T>,
	C::Id: Ord
{
	type Item = [C::Item; N];
	type Id = [C::Id; N];
	fn is_diff(&self) -> bool {
		self.iter().any(C::is_diff)
	}
	fn into_parts(self) -> (Self::Item, Self::Id) {
		let mut ids = [None; N];
		let mut iter = self.into_iter();
		let items = std::array::from_fn(|i| {
			let (item, id) = iter.next()
				.expect("should exist")
				.into_parts();
			ids[i] = Some(id);
			item
		});
		let mut ids = ids.map(Option::unwrap);
		ids.sort_unstable();
		(items, ids)
	}
}

impl<P, A> PredCombinatorCase<PredSingleComb<'static, P>> for (A,)
where
	P: PredParam<Id = A::Id>,
	A: PredCombinatorCase<P>,
{
	type Item = (A::Item,);
	type Id = (A::Id,);
	fn is_diff(&self) -> bool {
		self.0.is_diff()
	}
	fn into_parts(self) -> (Self::Item, Self::Id) {
		let (a,) = self;
		let (a_item, a_id) = a.into_parts();
		((a_item,), (a_id,))
	}
}

impl<P, Q, A, B> PredCombinatorCase<(P, Q)> for (A, B)
where
	P: PredParam<Id = A::Id>,
	Q: PredParam<Id = B::Id>,
	A: PredCombinatorCase<P>,
	B: PredCombinatorCase<Q>,
{
	type Item = (A::Item, B::Item);
	type Id = (A::Id, B::Id);
	fn is_diff(&self) -> bool {
		self.0.is_diff() || self.1.is_diff()
	}
	fn into_parts(self) -> (Self::Item, Self::Id) {
		let (a, b) = self;
		let (a_item, a_id) = a.into_parts();
		let (b_item, b_id) = b.into_parts();
		((a_item, b_item), (a_id, b_id))
	}
}

/// An item & ID pair of a `PredParam`, with their updated state.
pub enum PredCombCase<P, I> {
	Diff(P, I),
	Same(P, I),
}

impl<P, I> Clone for PredCombCase<P, I>
where
	P: PredItem,
	I: PredId,
{
	fn clone(&self) -> Self {
		match self {
			Self::Diff(item, id) => Self::Diff(item.clone(), *id),
			Self::Same(item, id) => Self::Same(item.clone(), *id),
		}
	}
}

/// ...
pub struct PredSingleComb<'w, A, K = CombNone>
where
	A: PredParam,
	K: CombKind,
{
	comb: A::Comb<'w, K>,
}

impl<'w, A, K> PredSingleComb<'w, A, K>
where
	A: PredParam,
	K: CombKind,
{
	pub(crate) fn new(comb: A::Comb<'w, K>) -> Self {
		Self { comb }
	}
}

impl<'w, A, K> Clone for PredSingleComb<'w, A, K>
where
	A: PredParam,
	K: CombKind,
{
	fn clone(&self) -> Self {
		Self {
			comb: self.comb.clone(),
		}
	}
}

impl<'w, A, K> IntoIterator for PredSingleComb<'w, A, K>
where
	A: PredParam,
	K: CombKind,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = PredSingleCombIter<'w, A, K>;
	fn into_iter(self) -> Self::IntoIter {
		PredSingleCombIter {
			iter: self.comb.into_iter(),
		}
	}
}

/// ...
pub struct PredSingleCombIter<'w, A, K>
where
	A: PredParam,
	K: CombKind,
{
	iter: <A::Comb<'w, K> as IntoIterator>::IntoIter,
}

impl<'w, A, K> Iterator for PredSingleCombIter<'w, A, K>
where
	A: PredParam,
	K: CombKind,
{
	type Item = <PredSingleComb<'w, A, K> as PredCombinator<K>>::Case;
	fn next(&mut self) -> Option<Self::Item> {
		self.iter.next().map(|x| (x,))
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.iter.size_hint()
	}
}

/// Combinator for `PredParam` tuple implementation.
pub struct PredPairComb<'w, A, B, K = CombNone>
where
	A: PredParam,
	B: PredParam,
	K: CombKind,
{
	a_comb: A::Comb<'w, K>,
	b_comb: B::Comb<'w, K::Pal>,
	a_inv_comb: A::Comb<'w, K::Inv>,
	b_inv_comb: B::Comb<'w, <<K::Inv as CombKind>::Pal as CombKind>::Inv>,
	kind: K,
}

impl<'w, A, B, K> Clone for PredPairComb<'w, A, B, K>
where
	A: PredParam,
	B: PredParam,
	K: CombKind,
{
	fn clone(&self) -> Self {
		Self {
			a_comb: self.a_comb.clone(),
			b_comb: self.b_comb.clone(),
			a_inv_comb: self.a_inv_comb.clone(),
			b_inv_comb: self.b_inv_comb.clone(),
			kind: self.kind,
		}
	}
}

impl<'w, A, B, K> PredPairComb<'w, A, B, K>
where
	A: PredParam,
	B: PredParam,
	K: CombKind,
{
	pub(crate) fn new(
		a_comb: A::Comb<'w, K>,
		b_comb: B::Comb<'w, K::Pal>,
		a_inv_comb: A::Comb<'w, K::Inv>,
		b_inv_comb: B::Comb<'w, <<K::Inv as CombKind>::Pal as CombKind>::Inv>,
		kind: K,
	) -> Self {
		Self {
			a_comb,
			b_comb,
			a_inv_comb,
			b_inv_comb,
			kind,
		}
	}
}

impl<'w, A, B, K> IntoIterator for PredPairComb<'w, A, B, K>
where
	A: PredParam,
	B: PredParam,
	K: CombKind,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = PredPairCombIter<'w, A, B, K>;
	fn into_iter(self) -> Self::IntoIter {
		PredPairCombIter::primary_next(
			self.a_comb.into_iter(),
			self.b_comb,
			self.a_inv_comb,
			self.b_inv_comb,
		)
	}
}

/// Iterator for 2-tuple [`PredParam`] types.
pub enum PredPairCombIter<'w, A, B, K>
where
	A: PredParam,
	B: PredParam,
	K: CombKind,
{
	Empty,
	Primary {
		a_iter: <A::Comb<'w, K> as IntoIterator>::IntoIter,
		a_case: A::Case<'w>,
		b_comb: B::Comb<'w, K::Pal>,
		b_iter: <B::Comb<'w, K::Pal> as IntoIterator>::IntoIter,
		a_inv_comb: A::Comb<'w, K::Inv>,
		b_inv_comb: B::Comb<'w, <<K::Inv as CombKind>::Pal as CombKind>::Inv>,
	},
	Secondary {
		b_iter: <B::Comb<'w, <<K::Inv as CombKind>::Pal as CombKind>::Inv> as IntoIterator>::IntoIter,
		b_case: <B::Comb<'w, <<K::Inv as CombKind>::Pal as CombKind>::Inv> as IntoIterator>::Item,
		a_comb: A::Comb<'w, K::Inv>,
		a_iter: <A::Comb<'w, K::Inv> as IntoIterator>::IntoIter,
	},
}

impl<'w, A, B, K> PredPairCombIter<'w, A, B, K>
where
	A: PredParam,
	B: PredParam,
	K: CombKind,
{
	fn primary_next(
		mut a_iter: <A::Comb<'w, K> as IntoIterator>::IntoIter,
		b_comb: B::Comb<'w, K::Pal>,
		a_inv_comb: A::Comb<'w, K::Inv>,
		b_inv_comb: B::Comb<'w, <<K::Inv as CombKind>::Pal as CombKind>::Inv>,
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
		mut b_iter: <B::Comb<'w, <<K::Inv as CombKind>::Pal as CombKind>::Inv> as IntoIterator>::IntoIter,
		a_comb: A::Comb<'w, K::Inv>,
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

impl<'w, A, B, K> Iterator for PredPairCombIter<'w, A, B, K>
where
	A: PredParam,
	B: PredParam,
	K: CombKind,
{
	type Item = <PredPairComb<'w, A, B, K> as PredCombinator<K>>::Case;
	fn next(&mut self) -> Option<Self::Item> {
		// !!! Put A/B in order of ascending size to reduce redundancy.
		match std::mem::replace(self, Self::Empty) {
			Self::Empty => None,
			
			Self::Primary {
				a_iter, a_case, b_comb, mut b_iter, a_inv_comb, b_inv_comb
			} => {
				if let Some(b_case) = b_iter.next() {
					let case = Some((a_case.clone(), b_case));
					*self = Self::Primary {
						a_iter, a_case, b_comb, b_iter, a_inv_comb, b_inv_comb
					};
					return case
				}
				*self = Self::primary_next(a_iter, b_comb, a_inv_comb, b_inv_comb);
				self.next()
			}
			
			Self::Secondary { b_iter, b_case, a_comb, mut a_iter } => {
				if let Some(a_case) = a_iter.next() {
					let case = Some((a_case, b_case.clone()));
					*self = Self::Secondary { b_iter, b_case, a_comb, a_iter };
					return case
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
					min
						.checked_add(a_max.checked_mul(b_max)?)?
						.checked_add(a_inv_max.checked_mul(b_inv_max)?)
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
pub struct PredArrayComb<'w, C, const N: usize, K = CombNone>
where
	C: PredParam,
	K: CombKind,
{
	a_comb: C::Comb<'w, CombBranch<K::Pal, K>>,
	b_comb: C::Comb<'w, CombBranch<K::Pal, K>>,
	pub(crate) a_index: usize,
	pub(crate) b_index: usize,
	pub(crate) is_nested: bool,
	kind: K,
}

impl<'w, C, const N: usize, K> Clone for PredArrayComb<'w, C, N, K>
where
	C: PredParam,
	K: CombKind,
{
	fn clone(&self) -> Self {
		Self {
			a_comb: self.a_comb.clone(),
			b_comb: self.b_comb.clone(),
			a_index: self.a_index,
			b_index: self.b_index,
			is_nested: self.is_nested,
			kind: self.kind,
		}
	}
}

impl<'w, C, const N: usize, K> PredArrayComb<'w, C, N, K>
where
	C: PredParam,
	C::Id: Ord,
	K: CombKind,
{
	pub fn new(
		a_comb: C::Comb<'w, CombBranch<K::Pal, K>>,
		b_comb: C::Comb<'w, CombBranch<K::Pal, K>>,
		kind: K,
	) -> Self {
		Self {
			a_comb,
			b_comb,
			a_index: 0,
			b_index: 0,
			is_nested: false,
			kind,
		}
	}
}

impl<'w, C, const N: usize, K> IntoIterator for PredArrayComb<'w, C, N, K>
where
	C: PredParam,
	C::Id: Ord,
	K: CombKind,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = PredArrayCombIter<'w, C, N, K>;
	fn into_iter(self) -> Self::IntoIter {
		let a_comb = self.a_comb;
		let b_comb = self.b_comb;
		let mut a_index = self.a_index;
		let mut b_index = self.b_index;
		let mut layer = N;
		let is_nested = self.is_nested;
		let kind = self.kind;
		
		let iters = std::array::from_fn(|i| {
			if i == N-1 && layer == N && !is_nested {
				let mut iter = b_comb.clone().into_iter();
				if let Some(case) = iter.nth(b_index) {
					a_index += 1;
					b_index += 1;
					Some((iter, case, [a_index, b_index]))
				} else {
					None
				}
			} else {
				let mut iter = a_comb.clone().into_iter();
				if let Some(case) = iter.nth(a_index) {
					a_index += 1;
					if kind.has_state(case.is_diff()) {
						b_index += 1;
						layer = layer.min(i + 1);
					} else if i == N-1
						&& layer == N
						&& b_comb.clone().into_iter().nth(b_index).is_none()
					{
						return None
					}
					Some((iter, case, [a_index, b_index]))
				} else {
					None
				}
			}
		});
		
		let iters = if iters.iter().all(Option::is_some) {
			Some(iters.map(Option::unwrap))
		} else {
			None
		};
		
		PredArrayCombIter { a_comb, b_comb, iters, layer, is_nested, kind }
	}
}

/// Iterator for array of [`PredParam`] type.
pub struct PredArrayCombIter<'w, C, const N: usize, K>
where
	C: PredParam,
	K: CombKind,
{
	a_comb: C::Comb<'w, CombBranch<K::Pal, K>>,
	b_comb: C::Comb<'w, CombBranch<K::Pal, K>>,
	iters: Option<[(
		<C::Comb<'w, CombBranch<K::Pal, K>> as IntoIterator>::IntoIter,
		C::Case<'w>,
		[usize; 2],
	); N]>,
	layer: usize,
	is_nested: bool,
	kind: K,
}

impl<'w, C, const N: usize, K> PredArrayCombIter<'w, C, N, K>
where
	C: PredParam,
	C::Id: Ord,
	K: CombKind,
{
	fn step(&mut self, i: usize) {
		if let Some(iters) = &mut self.iters {
			let (iter, case, [a_index, b_index]) = &mut iters[i];
			if let Some(next_case) = iter.next() {
				*a_index += 1;
				if self.kind.has_state(next_case.is_diff()) {
					*b_index += 1;
					self.layer = self.layer.min(i + 1);
				}
				*case = next_case;
				return
			}
			
			 // Bottom Layer Exhausted - Finished:
			if i == 0 {
				self.iters = None;
				return
			}
			
			 // Step Sub-Layer:
			if self.layer >= i {
		        self.layer = N;
			}
			self.step(i - 1);
			
			 // Reset Current Layer:
			if let Some(iters) = &mut self.iters {
				let (.., [sub_a_index, sub_b_index]) = iters[i - 1];
				let (iter, _, [a_index, b_index]) = &mut iters[i];
				*a_index = sub_a_index;
				*b_index = sub_b_index;
				
				if i == N-1 && self.layer == N && !self.is_nested {
					*iter = self.b_comb.clone().into_iter();
					if *b_index != 0 && iter.nth(*b_index - 1).is_none() {
						self.iters = None;
						return
					}
				} else {
					*iter = self.a_comb.clone().into_iter();
					if *a_index != 0 && iter.nth(*a_index - 1).is_none() {
						self.iters = None;
						return
					}
				}
				
				self.step(i);
			}
		}
	}
}

impl<'w, C, const N: usize, K> Iterator for PredArrayCombIter<'w, C, N, K>
where
	K: CombKind,
	C: PredParam,
	C::Id: Ord,
{
	type Item = <PredArrayComb<'w, C, N, K> as PredCombinator<K>>::Case;
	fn next(&mut self) -> Option<Self::Item> {
		if let Some(iters) = &self.iters {
			let case = std::array::from_fn(|i| iters[i].1.clone());
			self.step(N-1);
			Some(case)
		} else {
			None
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		fn falling_factorial(n: usize, r: usize) -> usize {
			if n < r {
				return 0
			}
			((1+n-r)..=n).product()
		}
		
		if let Some(iters) = &self.iters {
			let tot = |a_remaining, b_remaining| {
				let mut tot = 0;
				let mut div = 1;
				for i in (0..N).rev() {
					let (.., [mut a_index, mut b_index]) = iters[i];
					if i == N-1 {
						a_index -= 1;
					}
					let mut remaining = a_remaining - a_index;
					let mut num = falling_factorial(remaining, N - i);
					if i < self.layer && !self.is_nested {
						if i == N-1 {
							b_index -= 1;
						}
						remaining -= b_remaining - b_index;
						num -= falling_factorial(remaining, N - i);
					}
					div *= N - i;
					tot += num / div;
					// https://www.desmos.com/calculator/l6jawvulhk
				}
				tot
			};
			
			let (a_min, a_max) = self.a_comb.clone().into_iter().size_hint();
			let (b_min, b_max) = self.b_comb.clone().into_iter().size_hint();
			
			(
				tot(a_min, b_min),
				a_max.and_then(|x| Some(tot(x, b_max?)))
			)
		} else {
			(0, Some(0))
		}
	}
}

/// ...
#[derive(Clone)]
pub struct PredIdComb<I> {
	iter: I,
}

impl<I> PredIdComb<I> {
	pub(crate) fn new(iter: I) -> Self {
		Self { iter }
	}
}

impl<I> IntoIterator for PredIdComb<I>
where
	I: IntoIterator,
	I::Item: PredId,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = PredIdCombIter<I::IntoIter>;
	fn into_iter(self) -> Self::IntoIter {
		PredIdCombIter {
			iter: self.iter.into_iter(),
		}
	}
}

/// ...
pub struct PredIdCombIter<I> {
	iter: I,
}

impl<I> Iterator for PredIdCombIter<I>
where
	I: Iterator,
	I::Item: PredId,
{
	type Item = PredCombCase<WithId<I::Item>, I::Item>;
	fn next(&mut self) -> Option<Self::Item> {
		self.iter.next()
			.map(|id| PredCombCase::Same(WithId(id), id))
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.iter.size_hint()
	}
}

/// ...
pub struct SinglePredComb2<'p, T, P, K>
where
	P: PredParam,
	K: CombKind,
{
	iter: <P::Comb<'p, K> as IntoIterator>::IntoIter,
	node: NodeWriter<'p, PredStateCase<P::Id, T>>,
}

impl<'p, T, P, K> SinglePredComb2<'p, T, P, K>
where
	P: PredParam,
	K: CombKind,
{
	pub fn new(state: PredSubState2<'p, T, Single<P>, K>) -> Self {
		let mut comb = state.comb;
		comb.outer_skip(state.index);
		let iter = comb.into_iter();
		state.node.reserve(4 * iter.size_hint().0.max(1));
		Self {
			iter,
			node: NodeWriter::new(state.node),
		}
	}
}

impl<'p, T, P, K> Iterator for SinglePredComb2<'p, T, P, K>
where
	P: PredParam,
	K: CombKind,
{
	type Item = (
		&'p mut PredStateCase<P::Id, T>,
		PredParamItem<'p, P>,
	);
	fn next(&mut self) -> Option<Self::Item> {
		self.iter.next().map(|case| {
			let (item, id) = case.into_parts();
			let case = self.node.write(PredStateCase::new(id));
			(case, item)
		})
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.iter.size_hint()
	}
}

/// ...
pub struct NestedPredComb2<'p, T, A, B, K>
where
	A: PredParam,
	B: PredBranch,
	K: CombKind,
{
	iter: <A::Comb<'p, K::Pal> as IntoIterator>::IntoIter,
	sub_comb: [B::CombSplit<'p, CombBranch<K::Pal, K>>; 2],
	node: NodeWriter<'p, (<A as PredParam>::Id, Node<B::Case<T>>)>,
	kind: K,
}

impl<'p, T, A, B, K> NestedPredComb2<'p, T, A, B, K>
where
	A: PredParam,
	B: PredBranch,
	K: CombKind,
{
	pub fn new(state: PredSubState2<'p, T, Nested<A, B>, K>) -> Self {
		let (mut comb, sub_comb) = state.comb;
		comb.outer_skip(state.index);
		let iter = comb.into_iter();
		state.node.reserve(4 * iter.size_hint().0.max(1));
		Self {
			iter,
			sub_comb,
			node: NodeWriter::new(state.node),
			kind: state.kind,
		}
	}
}

impl<'p, T, A, B, K> Iterator for NestedPredComb2<'p, T, A, B, K>
where
	A: PredParam,
	B: PredBranch,
	K: CombKind,
{
	type Item = (
		PredSubState2<'p, T, B, CombBranch<K::Pal, K>>,
		PredParamItem<'p, A>,
	);
	fn next(&mut self) -> Option<Self::Item> {
		self.iter.next().map(|case| {
			let kind;
			let ind = if self.kind.has_state(case.is_diff()) {
				kind = CombBranch::A(self.kind.pal());
				0
			} else {
				kind = CombBranch::B(self.kind);
				1
			};
			let (item, id) = case.into_parts();
			let (_, node) = self.node.write((id, Node::default()));
			let sub_state = PredSubState2::new(self.sub_comb[ind].clone(), node, kind);
			(sub_state, item)
		})
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.iter.size_hint()
	}
}

/// ...
pub struct NestedPermPredComb2<'p, T, A, B, K>
where
	A: PredParam,
	B: PredPermBranch,
	K: CombKind,
{
	iter: <A::Comb<'p, K> as IntoIterator>::IntoIter,
	sub_comb: [B::CombSplit<'p, CombBranch<K::Pal, K>>; 2],
	node: NodeWriter<'p, (<A as PredParam>::Id, Node<B::Case<T>>)>,
	kind: K,
	count: [usize; 2],
}

impl<'p, T, A, const N: usize, B, K> NestedPermPredComb2<'p, T, [A; N], B, K>
where
	A: PredParam,
	A::Id: Ord,
	B: PredPermBranch<Output = A>,
	K: CombKind,
{
	pub fn new(state: PredSubState2<'p, T, NestedPerm<[A; N], B>, K>) -> Self {
		let (mut comb, sub_comb) = state.comb;
		let count = [
			comb.a_comb.clone().into_iter().count().saturating_sub(B::depth()),
			comb.b_comb.clone().into_iter().count(),
		];
		comb.outer_skip(state.index);
		let iter = comb.into_iter();
		state.node.reserve(4 * iter.size_hint().0.max(1));
		Self {
			iter,
			sub_comb,
			node: NodeWriter::new(state.node),
			kind: state.kind,
			count,
		}
	}
}

impl<'p, T, A, const N: usize, B, K> Iterator
	for NestedPermPredComb2<'p, T, [A; N], B, K>
where
	A: PredParam,
	A::Id: Ord,
	B: PredPermBranch<Output = A>,
	K: CombKind,
{
	type Item = (
		PredSubState2<'p, T, B, CombBranch<K::Pal, K>>,
		PredParamItem<'p, [A; N]>,
	);
	fn next(&mut self) -> Option<Self::Item> {
		if let Some(iters) = &self.iter.iters {
			let (.., ind) = iters[N-1];
			if ind[0] > self.count[0] || ind[1] > self.count[1] {
				return None
			}
			self.iter.next().map(|case| {
				let kind;
				let sub_comb;
				let index;
				if self.kind.has_state(case.is_diff()) {
					kind = CombBranch::A(self.kind.pal());
					sub_comb = self.sub_comb[0].clone();
					index = [ind[0], ind[0]];
				} else {
					kind = CombBranch::B(self.kind);
					sub_comb = self.sub_comb[1].clone();
					index = ind;
				};
				let (item, id) = case.into_parts();
				let (_, node) = self.node.write((id, Node::default()));
				let mut sub_state = PredSubState2::new(sub_comb, node, kind);
				sub_state.index = index;
				(sub_state, item)
			})
		} else {
			None
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		if let Some(iters) = &self.iter.iters {
			let (.., ind) = iters[N-1];
			if ind[0] > self.count[0] || ind[1] > self.count[1] {
				return (0, Some(0))
			}
			(
				(self.count[1] - ind[1].saturating_sub(1)).saturating_sub(B::depth()),
				Some(self.count[0] - ind[0].saturating_sub(1)),
			)
		} else {
			(0, Some(0))
		}
	}
}