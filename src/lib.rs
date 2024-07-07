#![allow(clippy::type_complexity)]

// use bevy::{
// 	ecs::prelude::*,
// 	DefaultPlugins
// };

mod comb;
mod pred;
mod node;

use comb::*;
use pred::*;

pub use pred::{Etc, Fetch, In, PredFetch, WithId, Single, Nested, NestedPerm};
pub use comb::{FetchComb, PredArrayComb, PredSingleComb, PredPairComb, PredIdComb, Misc};

use std::cmp::Reverse;
use std::collections::{BinaryHeap, btree_map, BTreeMap, HashMap};
use std::hash::Hash;
use std::time::{Duration, Instant};

use bevy_app::{App, MainScheduleOrder, Plugin, PreUpdate};
use bevy_ecs::schedule::{Schedule, Schedules, ScheduleLabel};
use bevy_ecs::system;
use bevy_ecs::system::{IntoSystem, ReadOnlySystemParam, System, SystemParamItem};
use bevy_ecs::world::{Mut, World};
use bevy_time::Time;

use chime::Chime;
use chime::pred::Prediction;
use chime::time;
use chime::time::InclusiveTimeRanges;

type BlankSystem = fn();

/// Builder entry point for adding chime events to a [`World`].
pub trait AddChimeEvent {
	fn add_chime_events<P, B, A, I, F, BgnSys, EndSys, OutSys, BgnMarker, EndMarker, OutMarker>(
		&mut self,
		events: ChimeEventBuilder<P, B, A, I, F, BgnSys, EndSys, OutSys>,
	) -> &mut Self
	where
		P: Prediction,
		P::TimeRanges: Send + Sync + 'static,
		B: PredBranch,
		A: ReadOnlySystemParam + 'static,
		I: IntoInput<B::Input> + Clone + Send + Sync + 'static,
		F: PredFn<P, B, A> + Send + Sync + 'static,
		BgnSys: IntoChimeEventSystem<B::Id, BgnMarker> + Clone + Send + Sync + 'static,
		EndSys: IntoChimeEventSystem<B::Id, EndMarker> + Clone + Send + Sync + 'static,
		OutSys: IntoChimeEventSystem<B::Id, OutMarker> + Clone + Send + Sync + 'static,
		BgnMarker: ChimeSystemParamGroup<B::Id>,
		EndMarker: ChimeSystemParamGroup<B::Id>,
		OutMarker: ChimeSystemParamGroup<B::Id>,
	;
}

impl AddChimeEvent for App {
	fn add_chime_events<P, B, A, I, F, BgnSys, EndSys, OutSys, BgnMarker, EndMarker, OutMarker>(
		&mut self,
		events: ChimeEventBuilder<P, B, A, I, F, BgnSys, EndSys, OutSys>,
	) -> &mut Self
	where
		P: Prediction,
		P::TimeRanges: Send + Sync + 'static,
		B: PredBranch,
		A: ReadOnlySystemParam + 'static,
		I: IntoInput<B::Input> + Clone + Send + Sync + 'static,
		F: PredFn<P, B, A> + Send + Sync + 'static,
		BgnSys: IntoChimeEventSystem<B::Id, BgnMarker> + Clone + Send + Sync + 'static,
		EndSys: IntoChimeEventSystem<B::Id, EndMarker> + Clone + Send + Sync + 'static,
		OutSys: IntoChimeEventSystem<B::Id, OutMarker> + Clone + Send + Sync + 'static,
		BgnMarker: ChimeSystemParamGroup<B::Id>,
		EndMarker: ChimeSystemParamGroup<B::Id>,
		OutMarker: ChimeSystemParamGroup<B::Id>,
	{
		assert!(self.is_plugin_added::<ChimePlugin>());
		
		let ChimeEventBuilder {
			pred_sys,
			begin_sys,
			end_sys,
			outlier_sys,
			input,
			..
		} = events;
		
		assert!(begin_sys.is_some() || end_sys.is_some() || outlier_sys.is_some());
		
		let id = self.world_mut().resource_mut::<ChimeEventMap>()
			.setup_id::<B::Id, P::TimeRanges>();
		
		let mut state = system::SystemState::<(
			<B::AllParams as PredCombinator>::Param,
			A,
			system::Local<bool>,
		)>::new(
			self.world_mut()
		);
		
		let system = move |world: &mut World| {
			// !!! Cache this in a Local so it doesn't need to reallocate much.
			// Maybe chop off nodes that go proportionally underused.
			let mut node = node::Node::default();
			{
				let (state, misc, mut is_active) = state.get(world);
				if *is_active == false {
					*is_active = true;
					pred_sys.run(
						PredSubState2::new(
							B::comb_split(unsafe { std::mem::transmute(&state) }, input.clone().into_input(), CombAll),
							&mut node,
							CombAll,
						),
						misc
					);
				} else {
					pred_sys.run(
						PredSubState2::new(
							B::comb_split(unsafe { std::mem::transmute(&state) }, input.clone().into_input(), CombAnyTrue),
							&mut node,
							CombAnyTrue,
						),
						misc
					);
				}
			}
			let time = world.resource::<Time<Chime>>().elapsed();
			world.resource_scope::<ChimeEventMap, ()>(|world, mut event_map| {
				event_map.sched(
					PredNode2::<B, P> { inner: node },
					time,
					id,
					begin_sys.as_ref(),
					end_sys.as_ref(),
					outlier_sys.as_ref(),
					world,
				);
			});
		};
		
		self.world_mut().resource_mut::<Schedules>()
			.get_mut(ChimeSchedule).unwrap()
			.add_systems(system);
		
		self
	}
}

/// ...
struct PredRun<T, M>(T, std::marker::PhantomData<M>);

/// This is pretty much a no-op except for testing purposes.
pub trait PredFn<P, B, A>
where
	B: PredBranch,
	A: ReadOnlySystemParam,
{
	fn run<K: CombKind>(&self, state: PredSubState2<P, B, K>, param: SystemParamItem<A>);
	
	fn into_events(self) -> ChimeEventBuilder<P, B, A, B::Input, Self, BlankSystem, BlankSystem, BlankSystem>
	where
		Self: Sized,
		B::Input: Default + Send + Sync + 'static,
	{
		ChimeEventBuilder::new(self, B::Input::default())
	}
}

impl<T, M, P, B> PredFn<P, B, ()> for PredRun<T, M>
where
	B: PredBranch,
	T: PredCaseFn<P, B, M>,
{
	fn run<K: CombKind>(&self, state: PredSubState2<P, B, K>, _param: ()) {
		self.0.run(state);
	}
}

/// Specialized function used for predicting and scheduling events, functionally
/// similar to a read-only [`bevy_ecs::system::SystemParamFunction`].
pub trait PredCaseFn<P, B: PredBranch, M> {
	fn run<K: CombKind>(&self, input: PredSubState2<P, B, K>);
	
	fn into_events(self) -> ChimeEventBuilder<P, B, (), B::Input, impl PredFn<P, B, ()>, BlankSystem, BlankSystem, BlankSystem>
	where
		Self: Sized,
		<B as PredBranch>::Input: Default + Send + Sync,
	{
		ChimeEventBuilder::new(
			PredRun::<Self, M>(self, std::marker::PhantomData),
			B::Input::default(),
		)
	}
	
	fn into_events_with_input<I>(self, input: I)
		-> ChimeEventBuilder<P, B, (), I, impl PredFn<P, B, ()>, BlankSystem, BlankSystem, BlankSystem>
	where
		Self: Sized,
	{
		ChimeEventBuilder::new(
			PredRun::<Self, M>(self, std::marker::PhantomData),
			input,
		)
	}
}

impl<F, P> PredCaseFn<P, Single<EmptyComb>, ()> for F
where
	P: Prediction,
	F: Fn() -> P,
{
	fn run<K: CombKind>(&self, input: PredSubState2<P, Single<EmptyComb>, K>) {
		for (case, ()) in input {
			let pred = self();
			case.set(pred)
		}
	}
}

impl<F, P, A, T> PredCaseFn<P, Single<PredSingleComb<A>>, ()> for F
where
	// Specifying `A::Item` as a separate parameter permits type elision.
	P: Prediction,
	A: PredCombinator<Item_= T>,
	T: PredItem2<A>,
	F: Fn(T,) -> P,
{
	fn run<K: CombKind>(&self, input: PredSubState2<P, Single<PredSingleComb<A>>, K>) {
		for (case, (a,)) in input {
			let pred = self(a,);
			case.set(pred)
		}
	}
}

impl<F, P, A, B, T, U,> PredCaseFn<P, Single<PredPairComb<A, B>>, ()> for F
where
	// Specifying `A::Item` et al. as separate parameters permits type elision.
	P: Prediction,
	A: PredCombinator<Item_= T>,
	B: PredCombinator<Item_= U>,
	T: PredItem2<A>,
	U: PredItem2<B>,
	F: Fn(T, U,) -> P,
{
	fn run<K: CombKind>(&self, input: PredSubState2<P, Single<PredPairComb<A, B>>, K>) {
		for (case, (a, b,)) in input {
			let pred = self(a, b,);
			case.set(pred);
		}
	}
}

impl<F, P, A, T, M, SubF, SubA> PredCaseFn<P, Nested<PredSingleComb<A>, SubA>, (SubF, M)> for F
where
	// Specifying `A::Item` as a separate parameter permits type elision.
	P: Prediction,
	A: PredCombinator<Item_= T>,
	T: PredItem2<A>,
	SubA: PredBranch,
	SubF: PredCaseFn<P, SubA, M>,
	F: Fn(T,) -> SubF,
{
	fn run<K: CombKind>(&self, input: PredSubState2<P, Nested<PredSingleComb<A>, SubA>, K>) {
		for (state, (a,)) in input {
			let x = self(a,);
			x.run(state);
		}
	}
}

impl<F, P, A, B, T, U, M, SubF, SubA> PredCaseFn<P, Nested<PredPairComb<A, B>, SubA>, (SubF, M)> for F
where
	// Specifying `A::Item` et al. as separate parameters permits type elision.
	P: Prediction,
	A: PredCombinator<Item_= T>,
	B: PredCombinator<Item_= U>,
	T: PredItem2<A>,
	U: PredItem2<B>,
	SubA: PredBranch,
	SubF: PredCaseFn<P, SubA, M>,
	F: Fn(T, U,) -> SubF,
{
	fn run<K: CombKind>(&self, input: PredSubState2<P, Nested<PredPairComb<A, B>, SubA>, K>) {
		for (state, (a, b,)) in input {
			let x = self(a, b,);
			x.run(state);
		}
	}
}

/// Begin/end-type system for a chime event (object-safe).
pub trait ChimeEventSystem: System<In=(), Out=()> + Send + Sync {}

impl<T> ChimeEventSystem for T
where
	T: System<In=(), Out=()> + Send + Sync + Clone,
{}

/// ...
pub trait IntoChimeEventSystem<I: PredId, P: ChimeSystemParamGroup<I>> {
	fn into_chime_event_system(self, id: I) -> impl ChimeEventSystem;
}

macro_rules! impl_into_chime_event_system {
    ($($param:ident $(, $rest:ident)*)?) => {
		impl_into_chime_event_system!(@impl $($param $(, $rest)*)?);
		$(impl_into_chime_event_system!($($rest),*);)?
	};
	(@impl $($param:ident),*) => {
		impl<F, I, $($param,)*> IntoChimeEventSystem<I, ($($param,)*)> for F
		where
			I: PredId,
			$($param: ChimeSystemParam<I>, $param::Param: 'static,)*
			F: FnMut($($param,)*) + FnMut($($param::Item<'_, '_>,)*)
				+ Clone + Send + Sync + 'static,
		{
			#[allow(unused_variables)]
			fn into_chime_event_system(mut self, id: I) -> impl ChimeEventSystem {
				IntoSystem::into_system(move |
					param: system::StaticSystemParam<($($param::Param,)*)>,
					time: bevy_ecs::system::Res<Time<Chime>>,
				| {
					let ($($param,)*) = param.into_inner();
					self($($param::fetch_param($param, id, time.elapsed()),)*)
				})
			}
		}
    };
}

impl_into_chime_event_system!{
	_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15
}

/// Builder for inserting a chime event into a [`World`].  
pub struct ChimeEventBuilder<T, P, A, I, F, BgnSys, EndSys, OutSys>
where
	P: PredBranch,
	A: ReadOnlySystemParam,
	F: PredFn<T, P, A>,
{
	pred_sys: F,
	begin_sys: Option<BgnSys>,
	end_sys: Option<EndSys>,
	outlier_sys: Option<OutSys>,
	input: I,
	_param: std::marker::PhantomData<fn(PredSubState2<T, P, CombAnyTrue>, SystemParamItem<A>)>,
}

impl<U, P, A, I, F, BgnSys, EndSys, OutSys> ChimeEventBuilder<U, P, A, I, F, BgnSys, EndSys, OutSys>
where
	P: PredBranch,
	A: ReadOnlySystemParam,
	F: PredFn<U, P, A>,
{
	fn new(pred_sys: F, input: I) -> Self {
		ChimeEventBuilder {
			pred_sys,
			begin_sys: None,
			end_sys: None,
			outlier_sys: None,
			input,
			_param: std::marker::PhantomData,
		}
	}
	
	/// The system that runs when the event's prediction becomes active.
	pub fn on_begin<T, Marker>(self, sys: T) -> ChimeEventBuilder<U, P, A, I, F, T, EndSys, OutSys>
	where
		T: IntoChimeEventSystem<P::Id, Marker>,
		Marker: ChimeSystemParamGroup<P::Id>,
	{
		assert!(self.begin_sys.is_none(), "can't have >1 begin systems");
		ChimeEventBuilder {
			pred_sys: self.pred_sys,
			begin_sys: Some(sys),
			end_sys: self.end_sys,
			outlier_sys: self.outlier_sys,
			input: self.input,
			_param: std::marker::PhantomData,
		}
	}
	
	/// The system that runs when the event's prediction becomes inactive.
	pub fn on_end<T, Marker>(self, sys: T) -> ChimeEventBuilder<U, P, A, I, F, BgnSys, T, OutSys>
	where
		T: IntoChimeEventSystem<P::Id, Marker>,
		Marker: ChimeSystemParamGroup<P::Id>,
	{
		assert!(self.end_sys.is_none(), "can't have >1 end systems");
		ChimeEventBuilder {
			pred_sys: self.pred_sys,
			begin_sys: self.begin_sys,
			end_sys: Some(sys),
			outlier_sys: self.outlier_sys,
			input: self.input,
			_param: std::marker::PhantomData,
		}
	}
	
	/// The system that runs when the event's prediction repeats excessively.
	pub fn on_repeat<T, Marker>(self, sys: T) -> ChimeEventBuilder<U, P, A, I, F, BgnSys, EndSys, T>
	where
		T: IntoChimeEventSystem<P::Id, Marker>,
		Marker: ChimeSystemParamGroup<P::Id>,
	{
		assert!(self.outlier_sys.is_none(), "can't have >1 outlier systems");
		ChimeEventBuilder {
			pred_sys: self.pred_sys,
			begin_sys: self.begin_sys,
			end_sys: self.end_sys,
			outlier_sys: Some(sys),
			input: self.input,
			_param: std::marker::PhantomData,
		}
	}
}

const RECENT_TIME: Duration = Duration::from_millis(500);
const OLDER_TIME: Duration = Duration::from_secs(10);

/// Chime event scheduling handler.
pub struct ChimePlugin;

impl Plugin for ChimePlugin {
	fn build(&self, app: &mut App) {
		let mut schedule_order = app.world_mut()
			.resource_mut::<MainScheduleOrder>();
		schedule_order.insert_after(PreUpdate, ChimeUpdate);
		app.add_systems(ChimeUpdate, update);
		app.world_mut().insert_resource(Time::<Chime>::default());
		app.world_mut().insert_resource(ChimeEventMap::default());
		app.world_mut().add_schedule(Schedule::new(ChimeSchedule));
	}
}

/// Update schedule inserted after [`PreUpdate`] in [`MainScheduleOrder`].
/// 
/// Used for Chime's main update loop to guarantee that it runs between resource
/// compilation ([`PreUpdate`]) and general app logic ([`Update`]).
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ChimeUpdate;

fn update(world: &mut World) {
	world.schedule_scope(ChimeSchedule, |world, pred_schedule| {
		let start = Duration::ZERO;
		let end = Duration::MAX;
		let fast = 1;
		let slow = 1;
		let time = world.resource::<Time>().elapsed();
		let target_time = (start + time*fast/slow).min(end);
		chime_update(world, target_time, pred_schedule);
	});
}

fn chime_update(world: &mut World, time: Duration, pred_schedule: &mut Schedule) {
	let mut tot_a = Duration::ZERO;
	let mut tot_b = Duration::ZERO;
	let mut num = 0;
	let a_time = Instant::now();
	
	let can_can_print = false;
	let mut can_print = can_can_print;
	
	pred_schedule.run(world);
	
	while let Some(duration) = world.resource::<ChimeEventMap>().first_time() {
		if time >= duration {
			world.resource_mut::<Time<Chime>>()
				.advance_to(duration);
			
			if can_print {
				can_print = false;
				println!("Time: {:?}", duration);
			}
			
			let a_time = Instant::now();
			world.resource_scope(|world, mut event_maps: Mut<ChimeEventMap>| {
				event_maps.run_first(world)
			});
			tot_a += Instant::now().duration_since(a_time);
			
			 // Reschedule Events:
			let a_time = Instant::now();
			pred_schedule.run(world);
			tot_b += Instant::now().duration_since(a_time);
			
			num += 1;
		} else {
			break
		}
	}
	
	world.resource_mut::<Time<Chime>>()
		.advance_to(time);
	
	let b_time = Instant::now();
	if can_can_print && b_time.duration_since(a_time) > chime::time::MILLISEC {
		println!("lag at {time:?} ({num:?}): {:?}", b_time.duration_since(a_time));
		println!("  run: {:?}", tot_a);
		println!("  pred: {:?}", tot_b);
	}
}

/// An individually scheduled event, generally owned by an `EventMap`.
struct ChimeEvent<T> {
	time: std::sync::Arc<std::sync::Mutex<Duration>>,
	times: Option<InclusiveTimeRanges<T>>,
	next_time: Option<Duration>,
	next_end_time: Option<Duration>,
	curr_time: Option<Duration>,
	prev_time: Option<Duration>,
	begin_sys: Option<Box<dyn System<In=(), Out=()>>>,
	end_sys: Option<Box<dyn System<In=(), Out=()>>>,
	outlier_sys: Option<Box<dyn System<In=(), Out=()>>>,
	recent_times: BinaryHeap<Reverse<Duration>>,
	older_times: BinaryHeap<Reverse<Duration>>,
	is_active: bool,
}

impl<T> Default for ChimeEvent<T> {
	fn default() -> Self {
		ChimeEvent {
			time: std::sync::Arc::new(std::sync::Mutex::new(Duration::ZERO)),
			times: None,
			next_time: None,
			next_end_time: None,
			curr_time: None,
			prev_time: None,
			begin_sys: None,
			end_sys: None,
			outlier_sys: None,
			recent_times: BinaryHeap::new(),
			older_times: BinaryHeap::new(),
			is_active: false,
		}
	}
}

impl<T> ChimeEvent<T>
where
	T: time::TimeRanges
{
	fn next_time(&mut self) -> Option<Duration> {
		let next_time = if self.is_active {
			&mut self.next_end_time
		} else {
			&mut self.next_time
		};
		if next_time.is_some() {
			return std::mem::take(next_time)
		}
		if let Some(times) = self.times.as_mut() {
			if let Some((a, b)) = times.next() {
				self.next_time = Some(a);
				self.next_end_time = Some(b);
				return self.next_time()
			}
		}
		None
	}
}

/// A set of independent `EventMap` values.
#[derive(system::Resource, Default)]
struct ChimeEventMap {
	table: Vec<Box<dyn AnyEventMap + Send + Sync>>
}

impl ChimeEventMap {
	fn first_time(&self) -> Option<Duration> {
		let mut time = None;
		for event_map in &self.table {
			if let Some(t) = event_map.first_time() {
				if let Some(min) = time {
					if t < min {
						time = Some(t);
					}
				} else {
					time = Some(t);
				}
			}
		}
		time
	}
	
	/// Initializes an `EventMap` and returns its stored index.
	fn setup_id<I, T>(&mut self) -> usize
	where
		I: PredId,
		T: time::TimeRanges + Send + Sync + 'static,
	{
		self.table.push(Box::<EventMap<I, T>>::default());
		self.table.len() - 1
	}
	
	fn sched<T, I, BgnSys, EndSys, OutSys, BgnMarker, EndMarker, OutMarker>(
		&mut self,
		input: impl IntoIterator<Item = PredStateCase<I, T>>,
		pred_time: Duration,
		event_id: usize,
		begin_sys: Option<&BgnSys>,
		end_sys: Option<&EndSys>,
		outlier_sys: Option<&OutSys>,
		world: &mut World,
	)
	where
		I: PredId,
		T: Prediction,
		T::TimeRanges: 'static,
		BgnSys: IntoChimeEventSystem<I, BgnMarker> + Clone,
		EndSys: IntoChimeEventSystem<I, EndMarker> + Clone,
		OutSys: IntoChimeEventSystem<I, OutMarker> + Clone,
		BgnMarker: ChimeSystemParamGroup<I>,
		EndMarker: ChimeSystemParamGroup<I>,
		OutMarker: ChimeSystemParamGroup<I>,
	{
		let event_map = self.table.get_mut(event_id)
			.expect("id must be initialized with ChimeEventMap::setup_id")
			.as_any_mut()
			.downcast_mut::<EventMap<I, T::TimeRanges>>()
			.expect("should always work");
		
		for case in input {
			let (case_id, case_times) = case.into_parts();
			
			 // Fetch/Initialize Event:
			let event = event_map.events.entry(case_id).or_insert_with(|| {
				let mut event = ChimeEvent::<T::TimeRanges>::default();
				
				 // Initialize Systems:
				if let Some(sys) = begin_sys {
					let mut sys = sys.clone().into_chime_event_system(case_id);
					sys.initialize(world);
					event.begin_sys = Some(Box::new(sys));
				}
				if let Some(sys) = end_sys {
					let mut sys = sys.clone().into_chime_event_system(case_id);
					sys.initialize(world);
					event.end_sys = Some(Box::new(sys));
				}
				if let Some(sys) = outlier_sys {
					let mut sys = sys.clone().into_chime_event_system(case_id);
					sys.initialize(world);
					event.outlier_sys = Some(Box::new(sys));
				}
				
				event
			});
			event.times = case_times
				.map(Prediction::into_ranges)
				.map(time::TimeRanges::inclusive);
			
			 // Fetch Next Time:
			event.next_time = None;
			let prev_time = std::mem::take(&mut event.prev_time);
			let is_active = std::mem::take(&mut event.is_active);
			let mut next_time;
			loop {
				next_time = event.next_time();
				if let Some(t) = next_time {
					if t > pred_time || (t == pred_time && !is_active && prev_time != Some(t)) {
						break
					}
					event.prev_time = Some(t);
					event.is_active = !event.is_active;
				} else {
					break
				}
			}
			if event.is_active != is_active {
				event.is_active = is_active;
				if is_active {
					event.next_time = next_time;
				} else {
					event.next_end_time = next_time;
				}
				next_time = Some(pred_time);
			}
			
			 // Update Prediction:
			if next_time != event.curr_time {
				 // Remove Old Prediction:
				if let Some(time) = event.curr_time {
					if let btree_map::Entry::Occupied(mut entry)
						= event_map.times.entry(time)
					{
						let list = entry.get_mut();
						let pos = list.iter()
							.position(|id| *id == case_id)
							.expect("this should always work");
						list.swap_remove(pos);
						if list.is_empty() {
							entry.remove();
						}
					} else {
						unreachable!()
					}
				}
				
				 // Insert New Prediction:
				event.curr_time = next_time;
				if let Some(time) = next_time {
					let list = event_map.times.entry(time)
						.or_default();
					
					if is_active {
						list.push(case_id); // End events (run first)
					} else {
						list.insert(0, case_id); // Begin events (run last)
					}
					
					 // Update Time Used by `Pred` Parameters:
					*event.time.lock().expect("should be available") = time;
				}
			}
		}
	}
	
	/// Runs the first upcoming event among all `EventMap`s - it's legit.
	fn run_first(&mut self, world: &mut World) {
		// If an event A's system doesn't overlap with an event B's system, and
		// as long as either event doesn't overlap with a scheduler that might
		// reschedule the other event, it doesn't matter which runs first. 
		let mut next = None;
		let mut time = None;
		for (index, event_map) in self.table.iter().enumerate() {
			if let Some(t) = event_map.first_time() {
				if let Some(min) = time {
					if t < min {
						time = Some(t);
						next = Some(index);
					}
				} else {
					time = Some(t);
					next = Some(index);
				}
			}
		}
		if let Some(next) = next {
			self.table[next].run_first(world);
		}
	}
}

/// A set of events related to a common method of scheduling.
struct EventMap<K, T> {
	events: HashMap<K, ChimeEvent<T>>,
	
	/// Reverse time-to-event map for quickly rescheduling events.
	times: BTreeMap<Duration, Vec<K>>,
}

impl<K, T> Default for EventMap<K, T> {
	fn default() -> Self {
		EventMap {
			events: HashMap::new(),
			times: BTreeMap::new(),
		}
	}
}

impl<K, T> EventMap<K, T>
where
	K: PredId,
	T: time::TimeRanges,
{
	fn first_time(&self) -> Option<Duration> {
		if let Some((&duration, _)) = self.times.first_key_value() {
			Some(duration)
		} else {
			None
		}
	}
	
	/// Runs the first upcoming event.
	fn run_first(&mut self, world: &mut World) {
		let mut entry = self.times.first_entry()
			.expect("this should always work");
		
		let time = *entry.key();
		let list = entry.get_mut();
		let key = list.pop()
			.expect("this should always work");
		
		if list.is_empty() {
			entry.remove();
		}
		
		let event = self.events.get_mut(&key)
			.expect("this should always work");
		
		debug_assert_eq!(event.curr_time, Some(time));
		event.prev_time = Some(time);
		
		 // Queue Up Next Prediction:
		event.is_active = !event.is_active;
		event.curr_time = event.next_time();
		if let Some(t) = event.curr_time {
			self.times.entry(t)
				.or_default()
				.push(key);
		}
		
		 // Overall vs Recent Average:
		while let Some(Reverse(t)) = event.older_times.peek() {
			if time < *t {
				break
			}
			event.older_times.pop();
		}
		while let Some(Reverse(t)) = event.recent_times.peek() {
			if time < *t {
				break
			}
			event.older_times.push(Reverse(*t + OLDER_TIME));
			event.recent_times.pop();
		}
		event.recent_times.push(Reverse(time + RECENT_TIME));
		
		 // End Event:
		if !event.is_active {
			if let Some(sys) = &mut event.end_sys {
				sys.run((), world);
			}
			return
		}
		
		 // Ignore Rapidly Repeating Events:
		let new_avg = (event.recent_times.len() as f32) / RECENT_TIME.as_secs_f32();
		let old_avg = (event.older_times.len() as f32) / OLDER_TIME.as_secs_f32();
		// println!(
		// 	"> {:?} at {:?} (avg: {:?}/sec, recent: {:?}/sec)",
		// 	key,
		// 	time,
		// 	(old_avg * 100.).round() / 100.,
		// 	(new_avg * 100.).round() / 100.,
		// );
		const LIMIT: f32 = 100.;
		let min_avg = (500 >> event.older_times.len().min(16)) as f32;
		let is_outlier = new_avg > (old_avg * LIMIT).max(min_avg);
		if is_outlier {
			if let Some(sys) = &mut event.outlier_sys {
				sys.run((), world);
			} else {
				// ??? Ignore, crash, warning, etc.
				// ??? If ignored, clear the recent average?
				println!(
					"event {:?} is repeating >{}x more than normal at time {:?}\n\
					old avg: {:?}/s\n\
					new avg: {:?}/s",
					key,
					LIMIT,
					time,
					old_avg,
					new_avg,
				);
				if let Some(sys) = &mut event.begin_sys {
					sys.run((), world);
				}
			}
		}
		
		 // Begin Event:
		else if let Some(sys) = &mut event.begin_sys {
			sys.run((), world);
		}
	}
}

/// Trait object of `EventMap` for use in `ChimeEventMap`.
trait AnyEventMap {
	fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
	fn first_time(&self) -> Option<Duration>;
	fn run_first(&mut self, world: &mut World);
}

impl<K, T> AnyEventMap for EventMap<K, T>
where
	K: PredId,
	T: time::TimeRanges + 'static,
{
	fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
		self
	}
	fn first_time(&self) -> Option<Duration> {
		EventMap::first_time(self)
	}
	fn run_first(&mut self, world: &mut World) {
		EventMap::run_first(self, world);
	}
}

/// Bevy schedule for re-predicting & scheduling events.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct ChimeSchedule;