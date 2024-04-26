use std::rc::Rc;
use bevy_ecs::change_detection::{Ref, Res};
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::query::{ArchetypeFilter, QueryIter};
use bevy_ecs::system::{Query, Resource};
use chime::pred::Prediction;
use crate::node::NodeWriter;
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
pub trait CombKind: Copy {
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
	type Case: PredCombinatorCase<Id=Self::Id>;
	
	type Param: PredParam<Id=Self::Id>;
	
	type IntoKind<Kind: CombKind>:
		PredCombinator<Kind, Case=Self::Case, Id=Self::Id>;
	
	fn into_kind<Kind: CombKind>(self, kind: Kind) -> Self::IntoKind<Kind>;
}

impl<K: CombKind> PredCombinator<K> for EmptyComb<K> {
	type Id = ();
	type Case = PredCombCase<(), ()>;
	
	type Param = ();
	
	type IntoKind<Kind: CombKind> = EmptyComb<Kind>;
	
	fn into_kind<Kind: CombKind>(self, kind: Kind) -> Self::IntoKind<Kind> {
		EmptyComb::new(kind)
	}
}

impl<'w, R, K> PredCombinator<K> for ResComb<'w, R, K>
where
	K: CombKind,
	R: Resource,
{
	type Id = ();
	type Case = PredCombCase<Res<'w, R>, ()>;
	
	type Param = Res<'w, R>;
	
	type IntoKind<Kind: CombKind> = ResComb<'w, R, Kind>;
	
	fn into_kind<Kind: CombKind>(self, kind: Kind) -> Self::IntoKind<Kind> {
		ResComb::new(Res::clone(&self.inner), kind)
	}
}

impl<'w, T, F, K> PredCombinator<K> for QueryComb<'w, T, F, K>
where
	K: CombKind,
	T: Component,
	F: ArchetypeFilter + 'static,
{
	type Id = Entity;
	type Case = PredCombCase<&'w T, Entity>;
	
	type Param = Query<'w, 'w, &'w T, F>;
	
	type IntoKind<Kind: CombKind> = QueryComb<'w, T, F, Kind>;
	
	fn into_kind<Kind: CombKind>(self, new_kind: Kind) -> Self::IntoKind<Kind> {
		match self {
			QueryComb::Normal { inner, .. } => {
				QueryComb::new(inner, new_kind)
			},
			QueryComb::Cached { slice, kind } => {
				if kind.states() == new_kind.states() {
					QueryComb::Cached {
						slice,
						kind: new_kind,
					}
				} else {
					unimplemented!("I don't think this happens, prove me wrong")
				}
			},
		}
	}
}

impl<A, B, K> PredCombinator<K> for PredPairComb<A, B, K>
where
	K: CombKind,
	A: PredCombinator,
	B: PredCombinator,
{
	type Id = (A::Id, B::Id);
	type Case = (A::Case, B::Case);
	
	type Param = (A::Param, B::Param);
	
	type IntoKind<Kind: CombKind> = PredPairComb<A, B, Kind>;
	
	fn into_kind<Kind: CombKind>(self, kind: Kind) -> Self::IntoKind<Kind> {
		PredPairComb::new(self.a_comb, self.b_comb, kind)
	}
}

impl<C, const N: usize, K> PredCombinator<K> for PredArrayComb<C, N, K>
where
	K: CombKind,
	C: PredCombinator,
	C::Id: Ord,
{
	type Id = [C::Id; N];
	type Case = [C::Case; N];
	
	type Param = [C::Param; N];
	
	type IntoKind<Kind: CombKind> = PredArrayComb<C, N, Kind>;
	
	fn into_kind<Kind: CombKind>(self, kind: Kind) -> Self::IntoKind<Kind> {
		if self.kind.pal().states() == kind.pal().states() {
			// Reusing `K::Pal` slice.
			PredArrayComb {
				comb: self.comb,
				slice: self.slice,
				index: self.index,
				min_diff_index: self.min_diff_index,
				min_same_index: self.min_same_index,
				max_diff_index: self.max_diff_index,
				max_same_index: self.max_same_index,
				kind,
			}
		} else {
			PredArrayComb::new(self.comb, kind)
		}
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
	
	type IntoKind<Kind: CombKind> = PredIdComb<I>;
	
	fn into_kind<Kind: CombKind>(self, _kind: Kind) -> Self::IntoKind<Kind> {
		self
	}
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
pub enum QueryComb<'w, T, F, K = CombNone>
where
	T: Component,
	F: ArchetypeFilter + 'static,
{
	Normal {
		inner: &'w Query<'w, 'w, (Ref<'static, T>, Entity), F>,
		kind: K,
	},
	Cached {
		slice: Rc<[PredCombCase<&'w T, Entity>]>,
		kind: K,
	},
}

impl<'w, T, F, K> QueryComb<'w, T, F, K>
where
	T: Component,
	F: ArchetypeFilter + 'static,
	K: CombKind,
{
	pub(crate) fn new(inner: &'w Query<'w, 'w, (Ref<'static, T>, Entity), F>, kind: K) -> Self {
		let comb = Self::Normal { inner, kind };
		if kind.has_state(true) == kind.has_state(false) {
			comb
		} else {
			Self::Cached {
				slice: comb.into_iter().collect(),
				kind,
			}
		}
	}
}

impl<T, F, K> Clone for QueryComb<'_, T, F, K>
where
	T: Component,
	F: ArchetypeFilter + 'static,
	K: Clone,
{
	fn clone(&self) -> Self {
		match self {
			Self::Normal { inner, kind } => Self::Normal {
				inner: Clone::clone(inner),
				kind: kind.clone(),
			},
			Self::Cached { slice, kind } => Self::Cached {
				slice: Rc::clone(&slice),
				kind: kind.clone(),
			},
		}
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
	T: Component,
	F: ArchetypeFilter,
{
	Normal { 
		iter: CombIter<QueryIter<'w, 'w, (Ref<'static, T>, Entity), F>, K>,
	},
	Cached {
		slice: Rc<[PredCombCase<&'w T, Entity>]>,
		index: usize,
	},
}

impl<'w, T, F, K> Iterator for QueryCombIter<'w, T, F, K>
where
	T: Component,
	F: ArchetypeFilter,
	K: CombKind,
{
	type Item = PredCombCase<&'w T, Entity>;
	fn next(&mut self) -> Option<Self::Item> {
		match self {
			Self::Normal { iter } => iter.next(),
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
			Self::Normal { iter } => iter.nth(n),
			Self::Cached { index, .. } => {
				*index += n;
				self.next()
			},
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
pub trait PredCombinatorCase: Clone {
	type Item: PredItem;
	type Id: PredId;
	fn is_diff(&self) -> bool;
	fn into_parts(self) -> (Self::Item, Self::Id);
}

impl PredCombinatorCase for () {
	type Item = ();
	type Id = ();
	fn is_diff(&self) -> bool {
		true
	}
	fn into_parts(self) -> (Self::Item, Self::Id) {
		((), ())
	}
}

impl<P, I> PredCombinatorCase for PredCombCase<P, I>
where
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

impl<C: PredCombinatorCase, const N: usize> PredCombinatorCase for [C; N]
where
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

/// Combinator for `PredParam` tuple implementation.
#[derive(Clone)]
pub struct PredPairComb<A, B, K = CombNone> {
	a_comb: A,
	b_comb: B,
	kind: K,
}

impl<A, B, K> PredPairComb<A, B, K> {
	pub(crate) fn new(a_comb: A, b_comb: B, kind: K) -> Self {
		Self {
			a_comb,
			b_comb,
			kind,
		}
	}
}

impl<A, B, K> IntoIterator for PredPairComb<A, B, K>
where
	A: PredCombinator,
	B: PredCombinator,
	K: CombKind,
{
	type Item = <Self::IntoIter as Iterator>::Item;
	type IntoIter = PredPairCombIter<A, B, K>;
	fn into_iter(self) -> Self::IntoIter {
		let Self { a_comb, b_comb, .. } = self;
		let a_inv_comb = a_comb.clone().into_kind(self.kind.inv());
		let b_inv_comb = b_comb.clone().into_kind(self.kind.inv().pal().inv());
		PredPairCombIter::primary_next(
			a_comb.into_kind(self.kind).into_iter(),
			b_comb.into_kind(self.kind.pal()),
			a_inv_comb,
			b_inv_comb,
		)
	}
}

/// Iterator for 2-tuple [`PredParam`] types.
pub enum PredPairCombIter<A, B, K>
where
	A: PredCombinator,
	B: PredCombinator,
	K: CombKind,
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
	A: PredCombinator,
	B: PredCombinator,
	K: CombKind,
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
	A: PredCombinator,
	B: PredCombinator,
	K: CombKind,
{
	type Item = <PredPairComb<A, B, K> as PredCombinator<K>>::Case;
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
					std::cmp::max(
						// This may be inaccurate if a new `CombKind` is added.
						// It should work for `K=CombNone|All|AnyTrue|AllFalse`.
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

/// [`PredParamVec::Split`] type of 2-tuples.
pub struct PredPairCombSplit<A, B, K> {
	a_iter: A,
	b_comb: B,
	kind: K,
}

impl<A, B, K> PredPairCombSplit<A, B, K>
where
	K: CombKind,
{
	pub fn new<C>(comb: PredPairComb<C, B, K>) -> Self
	where
		C: PredCombinator,
		C::IntoKind<K::Pal>: IntoIterator<IntoIter=A>,
	{
		Self {
			a_iter: comb.a_comb.into_kind(comb.kind.pal()).into_iter(),
			b_comb: comb.b_comb,
			kind: comb.kind,
		}
	}
}

impl<A, B, K> Iterator for PredPairCombSplit<A, B, K>
where
	A: Iterator,
	A::Item: PredCombinatorCase,
	B: PredCombinator,
	K: CombKind,
{
	type Item = (A::Item, B::IntoKind<CombBranch<
		K::Pal,
		<<K::Inv as CombKind>::Pal as CombKind>::Inv,
	>>);
	fn next(&mut self) -> Option<Self::Item> {
		if let Some(case) = self.a_iter.next() {
			// !!! Currently the main branch iterates over all A and if a case
			// is updated it iterates over all B, else it iterates over only the
			// updated B. To find the updated B cases it searches through *all*
			// B cases. I want to cache the updated B items in the latter case,
			// but not all B items in the former case.
			let kind = if case.is_diff() {
				CombBranch::A(self.kind.pal())
			} else {
				CombBranch::B(self.kind.inv().pal().inv())
			};
			let sub_comb = self.b_comb.clone().into_kind(kind);
			Some((case, sub_comb))
		} else {
			None
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.a_iter.size_hint()
	}
}

/// Combinator for `PredParam` array implementation.
pub struct PredArrayComb<C, const N: usize, K = CombNone>
where
	C: PredCombinator,
{
	comb: C,
	slice: Rc<[(C::Case, usize)]>,
	index: usize,
	min_diff_index: usize,
	min_same_index: usize,
	max_diff_index: usize,
	max_same_index: usize,
	kind: K,
}

impl<C, const N: usize, K> Clone for PredArrayComb<C, N, K>
where
	C: PredCombinator,
	K: Clone,
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
			kind: self.kind.clone(),
		}
	}
}

impl<C, const N: usize, K> PredArrayComb<C, N, K>
where
	C: PredCombinator,
	C::Id: Ord,
	K: CombKind,
{
	pub fn new(comb: C, kind: K) -> Self {
		let mut vec = comb.clone().into_kind::<K::Pal>(kind.pal()).into_iter()
			.map(|x| (x, usize::MAX))
			.collect::<Vec<_>>();
		
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
			kind,
		}
	}
}

impl<C, const N: usize, K> IntoIterator for PredArrayComb<C, N, K>
where
	K: CombKind,
	C: PredCombinator,
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
			kind: self.kind,
		};
		if N == 0 {
			return iter
		}
		
		 // Initialize Main Index:
		if match iter.kind.states() {
			[true, true] => false,
			[false, false] => true,
			[true, false] => iter.min_diff_index >= iter.slice.len(),
			[false, true] => iter.min_same_index >= iter.slice.len(),
		} {
			iter.index[N-1] = iter.slice.len();
		} else if iter.index[N-1] < iter.slice.len() {
			let (case, _) = &iter.slice[iter.index[N-1]];
			if self.kind.has_state(case.is_diff()) {
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
	C: PredCombinator,
{
	slice: Rc<[(C::Case, usize)]>,
	index: [usize; N],
	min_diff_index: usize,
	min_same_index: usize,
	layer: usize,
	kind: K,
}

impl<C, const N: usize, K> PredArrayCombIter<C, N, K>
where
	C: PredCombinator,
	C::Id: Ord,
	K: CombKind,
{
	fn step_index(&mut self, i: usize) -> bool {
		let index = self.index[i];
		if index >= self.slice.len() {
			return true
		}
		self.index[i] = match self.layer.cmp(&i) {
			std::cmp::Ordering::Equal => match self.kind.states() {
				[true, true] => index + 1,
				[false, false] => self.slice.len(),
				_ => {
					let (case, next_index) = &self.slice[index];
					
					 // Jump to Next Matching Case:
					if self.kind.has_state(case.is_diff()) {
						*next_index
					}
					
					 // Find Next Matching Case:
					else {
						let first_index = if self.kind.has_state(true) {
							self.min_diff_index
						} else {
							self.min_same_index
						};
						if index < first_index {
							first_index
						} else {
							let mut index = index + 1;
							while let Some((case, _)) = self.slice.get(index) {
								if self.kind.has_state(case.is_diff()) {
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
					if self.kind.has_state(case.is_diff()) {
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
				if !self.kind.has_all()
					&& self.slice[self.index[i + 1]].1 >= self.slice.len()
				{
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
	C: PredCombinator,
	C::Id: Ord,
{
	type Item = <PredArrayComb<C, N, K> as PredCombinator<K>>::Case;
	fn next(&mut self) -> Option<Self::Item> {
		if N == 0 || self.index[N-1] >= self.slice.len() {
			return None
		}
		let case = self.index.map(|i| self.slice[i].0.clone());
		self.step(0);
		Some(case)
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		// Currently always produces an exact size.
		
		if N == 0
			|| self.index[N-1] >= self.slice.len()
			|| self.kind.has_none()
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
			if i >= self.layer && !self.kind.has_all() {
				let first_index = if self.kind.has_state(true) {
					self.min_diff_index
				} else {
					self.min_same_index
				};
				while index < self.slice.len() {
					let (case, next_index) = &self.slice[index];
					index = if self.kind.has_state(case.is_diff()) {
						remaining -= 1;
						*next_index
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

/// [`PredParamVec::Split`] type of arrays.
pub struct PredArrayCombSplit<C, const N: usize, K>
where
	C: PredCombinator,
{
	inner: PredArrayComb<C, N, K>,
}

impl<C, const N: usize, K> PredArrayCombSplit<C, N, K>
where
	C: PredCombinator,
{
	pub fn new<const M: usize>(comb: PredArrayComb<C, M, K>) -> Self {
		Self {
			inner: PredArrayComb {
				comb: comb.comb,
				slice: comb.slice,
				index: comb.index,
				min_diff_index: comb.min_diff_index,
				min_same_index: comb.min_same_index,
				max_diff_index: comb.max_diff_index,
				max_same_index: comb.max_same_index,
				kind: comb.kind,
			}
		}
	}
}

impl<C, const N: usize, K> Iterator for PredArrayCombSplit<C, N, K>
where
	K: CombKind,
	C: PredCombinator,
	C::Id: Ord,
{
	type Item = (C::Case, PredArrayComb<C, N, CombBranch<
		K::Pal,
		<<K::Inv as CombKind>::Pal as CombKind>::Inv,
	>>);
	fn next(&mut self) -> Option<Self::Item> {
		if let Some(mut max_index) = self.inner.slice.len().checked_sub(N) {
			max_index = max_index.min(match self.inner.kind.states() {
				[true, true] => self.inner.slice.len(),
				[false, false] => 0,
				[true, false] => self.inner.max_diff_index,
				[false, true] => self.inner.max_same_index,
			});
			if self.inner.index >= max_index {
				return None
			}
		} else {
			return None
		};
		if let Some((case, _)) = self.inner.slice.get(self.inner.index) {
			self.inner.index += 1;
			let kind = if case.is_diff() {
				CombBranch::A(self.inner.kind.pal())
			} else {
				CombBranch::B(self.inner.kind.inv().pal().inv())
			};
			let sub_comb = self.inner.clone().into_kind(kind);
			Some((case.clone(), sub_comb))
		} else {
			None
		}
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		if let Some(mut max_index) = self.inner.slice.len().checked_sub(N) {
			max_index = max_index.min(match self.inner.kind.states() {
				[true, true] => self.inner.slice.len(),
				[false, false] => 0,
				[true, false] => self.inner.max_diff_index,
				[false, true] => self.inner.max_same_index,
			});
			let num = max_index - self.inner.index;
			(num, Some(num))
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

/// Produces all case combinations in need of a new prediction, alongside a
/// [`PredStateCase`] for scheduling.
pub struct PredComb<'p, T, P, K>
where
	P: PredParam,
	K: CombKind,
{
	iter: <<P::Comb<'p> as PredCombinator>::IntoKind<K> as IntoIterator>::IntoIter,
	node: NodeWriter<'p, PredStateCase<P::Id, T>>,
}

impl<'p, T, P, K> PredComb<'p, T, P, K>
where
	P: PredParam,
	K: CombKind,
{
	pub fn new<'s: 'p>(state: PredSubState<'p, 's, T, P, K>) -> Self {
		let iter = state.comb.into_iter();
		let node = state.node.init_data(4 * iter.size_hint().0.max(1));
		Self { iter, node }
	}
}

impl<'p, T, P, K> Iterator for PredComb<'p, T, P, K>
where
	T: Prediction,
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
			(
				self.node.write(PredStateCase::new(id)),
				item,
			)
		})
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.iter.size_hint()
	}
}