// use bevy::{
// 	ecs::prelude::*,
// 	DefaultPlugins
// };

mod pred;
mod node;

use pred::*;
use node::*;

pub use pred::{PredState, PredQuery};

use std::cmp::Reverse;
use std::collections::{BinaryHeap, btree_map, BTreeMap, HashMap};
use std::hash::Hash;
use std::time::{Duration, Instant};

use bevy_app::{App, Plugin, Update};
use bevy_ecs::schedule::{Schedule, Schedules, ScheduleLabel};
use bevy_ecs::system;
use bevy_ecs::system::{IntoSystem, ReadOnlySystemParam, System, SystemParamItem};
use bevy_ecs::world::{Mut, World};
use bevy_time::Time;

use chime::time::TimeRanges;

/// Builder entry point for adding chime events to a [`World`].
pub trait AddChimeEvent {
	fn add_chime_events<P, M, A, F>(&mut self, events: ChimeEventBuilder<P, M, A, F>) -> &mut Self
	where
		P: PredParam + 'static,
		M: PredId,
		A: ReadOnlySystemParam + 'static,
		F: PredFn<P, M, A> + Send + Sync + 'static;
}

impl AddChimeEvent for App {
	fn add_chime_events<P, M, A, F>(&mut self, events: ChimeEventBuilder<P, M, A, F>) -> &mut Self
	where
		P: PredParam + 'static,
		M: PredId,
		A: ReadOnlySystemParam + 'static,
		F: PredFn<P, M, A> + Send + Sync + 'static,
	{
		assert!(self.is_plugin_added::<ChimePlugin>());
		
		let ChimeEventBuilder {
			pred_sys,
			begin_sys,
			end_sys,
			outlier_sys,
			misc_state,
			..
		} = events;
		
		assert!(begin_sys.is_some() || end_sys.is_some() || outlier_sys.is_some());
		
		let id = self.world.resource_mut::<ChimeEventMap>()
			.setup_id::<PredParamId<P>>();
		
		let mut state = system::SystemState::<(P::Param, A)>::new(
			&mut self.world
		);
		
		let system = move |world: &mut World| {
			// !!! Cache this in a Local so it doesn't need to reallocate much.
			// Maybe chop off nodes that go proportionally underused.
			let mut node = Node::default();
			{
				let (state, misc) = state.get(world);
				pred_sys(
					PredState::new(&state, misc_state.clone(), &mut node),
					misc
				);
			}
			let time = world.resource::<Time>().elapsed();
			world.resource_scope::<ChimeEventMap, ()>(|world, mut event_map| {
				event_map.sched(
					node,
					time,
					id,
					begin_sys.as_ref().map(|x| x.as_ref()),
					end_sys.as_ref().map(|x| x.as_ref()),
					outlier_sys.as_ref().map(|x| x.as_ref()),
					world,
				);
			});
		};
		
		self.world.resource_mut::<Schedules>()
			.get_mut(ChimeSchedule).unwrap()
			.add_systems(system);
		
		self
	}
}

/// Specialized function used for predicting and scheduling events, functionally
/// similar to a read-only [`bevy_ecs::system::SystemParamFunction`].
pub trait PredFn<P: PredParam, M: PredId, A: ReadOnlySystemParam>:
	/*Fn(PredState<P>, A) + */ Fn(PredState<P, M>, SystemParamItem<A>)
{}

impl<P, M, A, F> PredFn<P, M, A> for F
where
	P: PredParam,
	M: PredId,
	A: ReadOnlySystemParam,
	F: /*Fn(PredState<P>, A) + */ Fn(PredState<P, M>, SystemParamItem<A>),
{}

/// Types that can be converted into a [`PredFn`].
pub trait IntoPredFn<P: PredParam, M: PredId, A: ReadOnlySystemParam>: Sized {
	fn into_pred_fn(self) -> impl PredFn<P, M, A>;
	
	fn into_events(self) -> ChimeEventBuilder<P, M, A, impl PredFn<P, M, A>>
	where
		(): std::borrow::Borrow<M>,
	{
		ChimeEventBuilder::new(
			self.into_pred_fn(),
			std::iter::once(*std::borrow::Borrow::borrow(&()))
		)
	}
	
	fn into_events_with_id(self, id: impl IntoIterator<Item=M>)
		-> ChimeEventBuilder<P, M, A, impl PredFn<P, M, A>>
	{
		ChimeEventBuilder::new(self.into_pred_fn(), id)
	}
}

macro_rules! impl_into_pred_fn {
	(@all $($param:ident $(, $rest:ident)*)?) => {
		impl_into_pred_fn!($($param $(, $rest)*)?);
		$(impl_into_pred_fn!(@all $($rest),*);)?
	};
	($($param:ident),*) => {
		impl<F, P, M, $($param: ReadOnlySystemParam),*> IntoPredFn<P, M, ($($param,)*)> for F
		where
			F: Fn(PredState<P, M>, $($param),*)
				+ Fn(PredState<P, M>, $(SystemParamItem<$param>),*),
			P: PredParam,
			M: PredId,
		{
			fn into_pred_fn(self) -> impl PredFn<P, M, ($($param,)*)> {
				move |state, misc| {
					let ($($param,)*) = misc;
					self(state, $($param),*);
				}
			}
		}
	}
}

impl_into_pred_fn!(@all
	_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15
);

/// Begin/end-type system for a chime event (object-safe).
trait ChimeEventSystem<I: PredId, M: PredId>: System<In=(), Out=()> + Send + Sync {
	fn init_sys(&self, store: &mut Option<Box<dyn System<In=(), Out=()>>>, world: &mut World);
}

impl<I, M, T> ChimeEventSystem<I, M> for T
where
	I: PredId,
	M: PredId,
	T: System<In=(), Out=()> + Send + Sync + Clone,
{
	fn init_sys(&self, store: &mut Option<Box<dyn System<In=(), Out=()>>>, world: &mut World) {
		let mut sys = self.clone();
		sys.initialize(world);
		*store = Some(Box::new(sys));
	}
}

/// Builder for inserting a chime event into a [`World`].  
pub struct ChimeEventBuilder<P, M, A, F>
where
	P: PredParam,
	M: PredId,
	A: ReadOnlySystemParam,
	F: PredFn<P, M, A>,
{
	pred_sys: F,
	begin_sys: Option<Box<dyn for<'a> ChimeEventSystem<PredParamId<'a, P>, M>>>,
	end_sys: Option<Box<dyn for<'a> ChimeEventSystem<PredParamId<'a, P>, M>>>,
	outlier_sys: Option<Box<dyn for<'a> ChimeEventSystem<PredParamId<'a, P>, M>>>,
	misc_state: Box<[M]>,
	_param: std::marker::PhantomData<fn(A)>,
}

impl<P, M, A, F> ChimeEventBuilder<P, M, A, F>
where
	P: PredParam,
	M: PredId,
	A: ReadOnlySystemParam,
	F: PredFn<P, M, A>,
{
	fn new(pred_sys: F, id: impl IntoIterator<Item=M>) -> Self {
		ChimeEventBuilder {
			pred_sys,
			begin_sys: None,
			end_sys: None,
			outlier_sys: None,
			misc_state: id.into_iter().collect(),
			_param: std::marker::PhantomData,
		}
	}
	
	/// The system that runs when the event's prediction becomes active.
	pub fn on_begin<T, Marker>(mut self, sys: T) -> Self
	where
		T: IntoSystem<(), (), Marker>,
		T::System: Send + Sync + Clone,
	{
		assert!(self.begin_sys.is_none(), "can't have >1 begin systems");
		self.begin_sys = Some(Box::new(IntoSystem::into_system(sys)));
		self
	}
	
	/// The system that runs when the event's prediction becomes inactive.
	pub fn on_end<T, Marker>(mut self, sys: T) -> Self
	where
		T: IntoSystem<(), (), Marker>,
		T::System: Send + Sync + Clone,
	{
		assert!(self.end_sys.is_none(), "can't have >1 end systems");
		self.end_sys = Some(Box::new(IntoSystem::into_system(sys)));
		self
	}
	
	/// The system that runs when the event's prediction repeats excessively.
	pub fn on_repeat<T, Marker>(mut self, sys: T) -> Self
	where
		T: IntoSystem<(), (), Marker>,
		T::System: Send + Sync + Clone,
	{
		assert!(self.outlier_sys.is_none(), "can't have >1 outlier systems");
		self.outlier_sys = Some(Box::new(IntoSystem::into_system(sys)));
		self
	}
}

const RECENT_TIME: Duration = Duration::from_millis(500);
const OLDER_TIME: Duration = Duration::from_secs(10);

/// Chime event scheduling handler.
pub struct ChimePlugin;

impl Plugin for ChimePlugin {
	fn build(&self, app: &mut App) {
		app.add_systems(Update, update);
		app.world.insert_resource(Time::<Chime>::default());
		app.world.insert_resource(ChimeEventMap::default());
		app.world.add_schedule(Schedule::new(ChimeSchedule));
	}
}

fn update(world: &mut World) {
	world.resource_scope(|world, old_time: Mut<Time>| {
	world.schedule_scope(ChimeSchedule, |world, pred_schedule| {
		let chime_time = world
			.resource::<Time<Chime>>()
			.as_generic();
		
		world.insert_resource(chime_time);
		
		let start = Duration::ZERO;
		let end = Duration::MAX;
		let fast = 1;
		let slow = 1;
		let time = (start + old_time.elapsed()*fast/slow).min(end);
		chime_update(world, time, pred_schedule);
		
		world.remove_resource::<Time>();
		world.resource_mut::<Time<Chime>>().advance_to(time);
	});
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
			world.resource_mut::<Time>().advance_to(duration);
			
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
	
	let b_time = Instant::now();
	if can_can_print && b_time.duration_since(a_time) > chime::time::MILLISEC {
		println!("lag at {time:?} ({num:?}): {:?}", b_time.duration_since(a_time));
		println!("  run: {:?}", tot_a);
		println!("  pred: {:?}", tot_b);
	}
}

/// An individually scheduled event, generally owned by an `EventMap`.
struct ChimeEvent {
	times: Box<dyn Iterator<Item = (Duration, Duration)> + Send + Sync>,
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

impl Default for ChimeEvent {
	fn default() -> Self {
		ChimeEvent {
			times: Box::new(std::iter::empty()),
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

impl ChimeEvent {
	fn next_time(&mut self) -> Option<Duration> {
		let next_time = if self.is_active {
			&mut self.next_end_time
		} else {
			&mut self.next_time
		};
		if next_time.is_some() {
			return std::mem::take(next_time)
		}
		if let Some((a, b)) = self.times.next() {
			self.next_time = Some(a);
			self.next_end_time = Some(b);
			self.next_time()
		} else {
			None
		}
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
	fn setup_id<I: PredId>(&mut self) -> usize {
		self.table.push(Box::<EventMap<I>>::default());
		self.table.len() - 1
	}
	
	fn sched<I: PredId, M: PredId>(
		&mut self,
		input: impl IntoIterator<Item=PredStateCase<I, M>>,
		pred_time: Duration,
		event_id: usize,
		begin_sys: Option<&dyn ChimeEventSystem<I, M>>,
		end_sys: Option<&dyn ChimeEventSystem<I, M>>,
		outlier_sys: Option<&dyn ChimeEventSystem<I, M>>,
		world: &mut World,
	) {
		let event_map = self.table.get_mut(event_id)
			.expect("id must be initialized with ChimeEventMap::setup_id")
			.as_any_mut()
			.downcast_mut::<EventMap<I>>()
			.expect("should always work");
		
		for case in input {
			let (case_id, case_times) = case.into_parts();
			
			 // Fetch/Initialize Event:
			let event = event_map.events.entry(case_id).or_insert_with(|| {
				let mut event = ChimeEvent::default();
				
				 // Initialize Systems:
				world.insert_resource(PredSystemId(Box::new(case_id)));
				if let Some(sys) = begin_sys {
					sys.init_sys(&mut event.begin_sys, world);
				}
				if let Some(sys) = end_sys {
					sys.init_sys(&mut event.end_sys, world);
				}
				if let Some(sys) = outlier_sys {
					sys.init_sys(&mut event.outlier_sys, world);
				}
				world.remove_resource::<PredSystemId>();
				
				event
			});
			event.times = case_times
				.unwrap_or_else(|| Box::new(TimeRanges::empty()));
			
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
struct EventMap<K> {
	events: HashMap<K, ChimeEvent>,
	
	/// Reverse time-to-event map for quickly rescheduling events.
	times: BTreeMap<Duration, Vec<K>>,
}

impl<K> Default for EventMap<K> {
	fn default() -> Self {
		EventMap {
			events: HashMap::new(),
			times: BTreeMap::new(),
		}
	}
}

impl<K: PredId> EventMap<K> {
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

impl<K: PredId> AnyEventMap for EventMap<K> {
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

/// Context for a `bevy::time::Time`.
#[derive(Default)]
pub struct Chime;