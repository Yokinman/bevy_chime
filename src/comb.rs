use std::marker::PhantomData;
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
///  AllSame  | AllSame  | AnyDiff  | AnyDiff
///  AnyTrue  | All      | AllFalse | AnyTrue
///  AnyFalse | All      | AllTrue  | AnyFalse
///  AnyDiff  | AllSame  | AllSame  | AnyDiff
/// ```
/// 
/// - `None`: No combinations.
/// - `All`: All possible combinations.
/// - `AllTrue`: Combinations where all are true.
/// - `AllFalse`: Combinations where all are false.
/// - `AllSame`: Combinations where all are the same as each other.
/// - `AnyTrue`: Combinations where at least one is true.
/// - `AnyFalse`: Combinations where at least one is false.
/// - `AnyDiff`: Combinations where at least one is different from the rest.
pub trait CombKind {
	type Pal: CombKind;
	type Inv: CombKind;
	
	fn has_state(state: bool) -> bool;
	
	#[inline]
	fn has_all() -> bool {
		Self::has_state(true) && Self::has_state(false)
	}
	
	#[inline]
	fn has_none() -> bool {
		!(Self::has_state(true) || Self::has_state(false))
	}
	
	#[inline]
	fn states() -> [bool; 2] {
		[Self::has_state(true), Self::has_state(false)]
	}
}

macro_rules! def_comb_kind {
	($name:ident, $pal:ty, $inv:ty, $state:pat => $has_state:expr) => {
		/// See [`CombKind`].
		pub struct $name;
		
		impl CombKind for $name {
			type Pal = $pal;
			type Inv = $inv;
			#[inline]
			fn has_state($state: bool) -> bool {
				$has_state
			}
		}
	};
}

            // Name,         Pal,          Inv,          Filter
def_comb_kind!(CombNone,     CombNone,     CombAll,      _ => false);
def_comb_kind!(CombAll,      CombAll,      CombNone,     _ => true);
def_comb_kind!(CombAllFalse, CombAllFalse, CombAnyTrue,  x => !x);
def_comb_kind!(CombAnyTrue,  CombAll,      CombAllFalse, x => x);

/// Combinator type produced by `PredParam::comb`.
pub trait PredCombinator<K: CombKind = CombNone>:
	Clone + IntoIterator<Item=Self::Case>
{
	type Id: PredId;
	type Case: PredCombinatorCase<Id=Self::Id>;
	
	type IntoKind<Kind: CombKind>:
		PredCombinator<Kind, Case=Self::Case, Id=Self::Id>;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind>;
}

impl<K: CombKind> PredCombinator<K> for EmptyComb<K> {
	type Id = ();
	type Case = PredCombCase<(), ()>;
	
	type IntoKind<Kind: CombKind> = EmptyComb<Kind>;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind> {
		EmptyComb {
			kind: PhantomData,
		}
	}
}

impl<'w, R, K> PredCombinator<K> for ResComb<'w, R, K>
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

impl<'w, T, F, K> PredCombinator<K> for QueryComb<'w, T, F, K>
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

impl<A, B, K> PredCombinator<K> for PredPairComb<A, B, K>
where
	K: CombKind,
	A: PredCombinator,
	B: PredCombinator,
{
	type Id = (A::Id, B::Id);
	type Case = (A::Case, B::Case);
	
	type IntoKind<Kind: CombKind> = PredPairComb<A, B, Kind>;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind> {
		PredPairComb::new(self.a_comb, self.b_comb)
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
	
	type IntoKind<Kind: CombKind> = PredArrayComb<C, N, Kind>;
	
	fn into_kind<Kind: CombKind>(self) -> Self::IntoKind<Kind> {
		if K::Pal::states() == Kind::Pal::states() {
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

/// Combinator for each [`PredParamVec::Split`] case.
pub enum PredSubComb<C: PredCombinator, K: CombKind> {
	Diff(C::IntoKind<K::Pal>),
	Same(C::IntoKind<<<K::Inv as CombKind>::Pal as CombKind>::Inv>),
}

/// Combinator for `PredParam` `()` implementation.
pub struct EmptyComb<K = CombNone> {
	kind: PhantomData<K>,
}

impl<K: CombKind> EmptyComb<K> {
	pub fn new() -> Self {
		Self {
			kind: PhantomData,
		}
	}
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

impl<'w, T, K> ResComb<'w, T, K>
where
	K: CombKind,
	T: Resource,
{
	pub fn new(inner: Res<'w, T>) -> Self {
		Self {
			inner,
			kind: PhantomData,
		}
	}
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

impl<'w, T, F, K> QueryComb<'w, T, F, K>
where
	K: CombKind,
	T: Component,
	F: ArchetypeFilter + 'static,
{
	pub fn new(inner: &'w Query<'w, 'w, (Ref<'static, T>, Entity), F>) -> Self {
		Self {
			inner,
			kind: PhantomData,
		}
	}
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
	pub fn new(iter: T) -> Self {
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
		if K::has_none() {
			return None
		}
		for (item, id) in self.iter.by_ref() {
			let is_updated = P::is_updated(&item);
			if K::has_state(is_updated) {
				return Some(if is_updated {
					PredCombCase::Diff(P::into_ref(item), id)
				} else {
					PredCombCase::Same(P::into_ref(item), id)
				})
			}
		}
		None
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		match K::states() {
			[false, false] => (0, Some(0)),
			[true, true] => self.iter.size_hint(),
			_ => (0, self.iter.size_hint().1)
		}
	}
}

/// Item of a [`PredCombinator`]'s iterator.
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
	A: PredCombinator,
	B: PredCombinator,
{
	pub fn new(a_comb: A, b_comb: B) -> Self {
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
	A: PredCombinator,
	B: PredCombinator,
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
	A: PredCombinator,
	B: PredCombinator,
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
	A: PredCombinator,
	B: PredCombinator,
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
	A: PredCombinator,
	B: PredCombinator,
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
	kind: PhantomData<K>,
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
			a_iter: comb.a_comb.into_kind().into_iter(),
			b_comb: comb.b_comb,
			kind: PhantomData,
		}
	}
}

impl<A, B, K> Iterator for PredPairCombSplit<A, B, K>
where
	K: CombKind,
	A: Iterator,
	A::Item: PredCombinatorCase,
	B: PredCombinator,
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
	kind: PhantomData<K>,
}

impl<C, const N: usize, K> Clone for PredArrayComb<C, N, K>
where
	C: PredCombinator,
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
	C: PredCombinator,
	C::Id: Ord,
{
	pub fn new(comb: C) -> Self {
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
			kind: PhantomData,
		};
		if N == 0 {
			return iter
		}
		
		 // Initialize Main Index:
		if match K::states() {
			[true, true] => false,
			[false, false] => true,
			[true, false] => iter.min_diff_index >= iter.slice.len(),
			[false, true] => iter.min_same_index >= iter.slice.len(),
		} {
			iter.index[N-1] = iter.slice.len();
		} else if iter.index[N-1] < iter.slice.len() {
			let (case, _) = iter.slice[iter.index[N-1]];
			if K::has_state(case.is_diff()) {
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
	kind: PhantomData<K>,
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
			std::cmp::Ordering::Equal => match K::states() {
				[true, true] => index + 1,
				[false, false] => self.slice.len(),
				_ => {
					let (case, next_index) = self.slice[index];
					
					 // Jump to Next Matching Case:
					if K::has_state(case.is_diff()) {
						next_index
					}
					
					 // Find Next Matching Case:
					else {
						let first_index = if K::has_state(true) {
							self.min_diff_index
						} else {
							self.min_same_index
						};
						if index < first_index {
							first_index
						} else {
							let mut index = index + 1;
							while let Some((case, _)) = self.slice.get(index) {
								if K::has_state(case.is_diff()) {
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
					if K::has_state(case.is_diff()) {
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
				if !K::has_all()
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
		let case = self.index.map(|i| self.slice[i].0);
		self.step(0);
		Some(case)
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		// Currently always produces an exact size.
		
		if N == 0
			|| self.index[N-1] >= self.slice.len()
			|| K::has_none()
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
			if i >= self.layer && !K::has_all() {
				let first_index = if K::has_state(true) {
					self.min_diff_index
				} else {
					self.min_same_index
				};
				while index < self.slice.len() {
					let (case, next_index) = self.slice[index];
					index = if K::has_state(case.is_diff()) {
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
				kind: PhantomData,
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
	type Item = (C::Case, PredSubComb<PredArrayComb<C, N>, K>);
	fn next(&mut self) -> Option<Self::Item> {
		if let Some(mut max_index) = self.inner.slice.len().checked_sub(N) {
			max_index = max_index.min(match K::states() {
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
			max_index = max_index.min(match K::states() {
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

/// Produces all case combinations in need of a new prediction, alongside a
/// [`PredStateCase`] for scheduling.
pub struct PredComb<'p, T, P, K>
where
	P: PredParam,
	K: CombKind,
{
	iter: <<P::Comb<'p> as PredCombinator>::IntoKind<K> as IntoIterator>::IntoIter,
	curr: Option<<<P::Comb<'p> as PredCombinator>::IntoKind<K> as IntoIterator>::Item>,
	node: NodeWriter<'p, PredStateCase<P::Id, T>>,
}

impl<'p, T, P, K> PredComb<'p, T, P, K>
where
	P: PredParam,
	K: CombKind,
{
	pub fn new<'s: 'p>(state: PredSubState<'p, 's, T, P, K>) -> Self {
		let mut iter = state.comb.into_iter();
		let node = state.node.init_data(4 * iter.size_hint().0.max(1));
		let curr = iter.next();
		Self { iter, curr, node }
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
		<PredParamItem<'p, P> as PredItem>::Ref,
	);
	fn next(&mut self) -> Option<Self::Item> {
		while let Some(case) = self.curr {
			let (item, id) = case.into_parts();
			self.curr = self.iter.next();
			return Some((
				self.node.write(PredStateCase::new(id)),
				item,
			))
		}
		None
	}
	fn size_hint(&self) -> (usize, Option<usize>) {
		if self.curr.is_some() {
			let (min, max) = self.iter.size_hint();
			(
				min + 1,
				max.and_then(|x| x.checked_add(1))
			)
		} else {
			(0, Some(0))
		}
	}
}

/// Iterator of [`PredSubStateSplit`].
pub enum PredCombSplit<'p, T, P, K>
where
	P: PredParam,
	K: CombKind,
{
	Diff(PredComb<'p, T, P, K::Pal>),
	Same(PredComb<'p, T, P, <<K::Inv as CombKind>::Pal as CombKind>::Inv>),
}

impl<'p, T, P, K> Iterator for PredCombSplit<'p, T, P, K>
where
	P: PredParam,
	K: CombKind,
	PredComb<'p, T, P, K::Pal>:
		Iterator,
	PredComb<'p, T, P, <<K::Inv as CombKind>::Pal as CombKind>::Inv>:
		Iterator<Item = <PredComb<'p, T, P, K::Pal> as Iterator>::Item>,
{
	type Item = <PredComb<'p, T, P, K::Pal> as Iterator>::Item;
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