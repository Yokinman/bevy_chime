// use bevy::{
// 	ecs::prelude::*,
// 	DefaultPlugins
// };

use std::cmp::Reverse;
use std::collections::{BinaryHeap, btree_map, BTreeMap, HashMap};
use std::hash::Hash;

use bevy::app::{App, Plugin, Update};
use bevy::ecs::entity::Entity;
use bevy::ecs::schedule::{Schedule, Schedules, ScheduleLabel};
use bevy::ecs::system::{In, IntoSystem, Res, ResMut, Resource, ReadOnlySystem, System};
use bevy::ecs::world::{Mut, World};

use std::time::{Duration, Instant};
use bevy::time::Time;
use chime::time::TimeRanges;

use bevy::input::{ButtonInput, keyboard::KeyCode};

/// Builder entry point for adding chime events to a [`World`].
pub trait AddChimeEvent {
	fn add_chime_events<P, S>(&mut self, events: ChimeEventBuilder<P, S>) -> &mut Self
	where
		P: PredHash + Send + Sync + 'static,
		S: ReadOnlySystem<In=PredState<P>, Out=PredState<P>>;
}

impl AddChimeEvent for App {
	fn add_chime_events<P, S>(&mut self, events: ChimeEventBuilder<P, S>) -> &mut Self
	where
		P: PredHash + Send + Sync + 'static,
		S: ReadOnlySystem<In=PredState<P>, Out=PredState<P>>,
	{
		assert!(self.is_plugin_added::<ChimePlugin>());
		
		let ChimeEventBuilder {
			pred_sys,
			begin_sys,
			end_sys,
			outlier_sys,
			..
		} = events;
		
		assert!(begin_sys.is_some() || end_sys.is_some() || outlier_sys.is_some());
		
		let id = self.world.resource_mut::<ChimeEventMap>().setup_id();
		
		let input = || -> PredState<P> {
			PredState(Vec::new())
		};
		
		let compile = move |In(state): In<PredState<P>>, mut pred: ResMut<ChimeEventMap>, time: Res<Time>| {
			pred.sched(
				state,
				time.elapsed(),
				id,
				begin_sys.as_ref().map(|x| x.as_ref()),
				end_sys.as_ref().map(|x| x.as_ref()),
				outlier_sys.as_ref().map(|x| x.as_ref()),
			);
		};
		
		let system = input.pipe(pred_sys).pipe(compile);
		
		self.world.resource_mut::<Schedules>()
			.get_mut(ChimeSchedule).unwrap()
			.add_systems(system);
		
		self
	}
}

/// Begin/end-type system for a chime event (object-safe).
trait ChimeEventSystem: System<Out=()> + Send + Sync {
	fn add_to_schedule(&self, schedule: &mut Schedule, input: Self::In);
}

impl<T: System<Out=()> + Send + Sync + Clone> ChimeEventSystem for T
where
	<T as System>::In: Send + Sync + Copy
{
	fn add_to_schedule(&self, schedule: &mut Schedule, input: Self::In) {
		let input_sys = move || -> Self::In {
			input
		};
		schedule.add_systems(input_sys.pipe(self.clone()));
	}
}

/// Builder for inserting a chime event into a [`World`].  
pub struct ChimeEventBuilder<P, S> {
	case: std::marker::PhantomData<P>,
	pred_sys: S,
	begin_sys: Option<Box<dyn ChimeEventSystem<In=P, Out=()>>>,
	end_sys: Option<Box<dyn ChimeEventSystem<In=P, Out=()>>>,
	outlier_sys: Option<Box<dyn ChimeEventSystem<In=P, Out=()>>>,
}

impl<P, S> ChimeEventBuilder<P, S>
where
	P: PredHash + Send + Sync + 'static,
	S: ReadOnlySystem<In=PredState<P>, Out=PredState<P>>,
{
	pub fn new<M>(pred_sys: impl IntoSystem<S::In, S::Out, M, System=S>) -> Self {
		Self {
			case: std::marker::PhantomData,
			pred_sys: IntoSystem::into_system(pred_sys),
			begin_sys: None,
			end_sys: None,
			outlier_sys: None,
		}
	}
	
	/// The system that runs when the event's prediction becomes active.
	pub fn on_begin<T, M>(mut self, sys: T) -> Self
	where
		T: IntoSystem<P, (), M>,
		T::System: Send + Sync + Clone,
	{
		assert!(self.begin_sys.is_none(), "can't have >1 begin systems");
		self.begin_sys = Some(Box::new(IntoSystem::into_system(sys)));
		self
	}
	
	/// The system that runs when the event's prediction becomes inactive.
	pub fn on_end<T, M>(mut self, sys: T) -> Self
	where
		T: IntoSystem<P, (), M>,
		T::System: Send + Sync + Clone,
	{
		assert!(self.end_sys.is_none(), "can't have >1 end systems");
		self.end_sys = Some(Box::new(IntoSystem::into_system(sys)));
		self
	}
	
	/// The system that runs when the event's prediction repeats excessively.
	pub fn on_repeat<T, M>(mut self, sys: T) -> Self
	where
		T: IntoSystem<P, (), M>,
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
	
	let can_can_print = world.resource::<ButtonInput<KeyCode>>().pressed(KeyCode::Space);
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
			world.resource_scope(|world, mut pred: Mut<ChimeEventMap>| {
				let key = pred.pop();
				let event = pred.table
					.get_mut(key.0).unwrap()
					.get_mut(&key.1).unwrap();
				
				if !event.is_active {
					event.end_schedule.run(world);
					return
				}
				// let last_time = std::mem::replace(&mut event.last_time, Some(duration));
				
				 // Ignore Rapidly Repeating Events:
				let new_avg = (event.recent_times.len() as f32) / RECENT_TIME.as_secs_f32();
				let old_avg = (event.older_times.len() as f32) / OLDER_TIME.as_secs_f32();
				if can_can_print {
					println!(
						"> {:?} at {:?} (avg: {:?}/sec, recent: {:?}/sec)",
						key,
						duration,
						(old_avg * 100.).round() / 100.,
						(new_avg * 100.).round() / 100.,
					);
				}
				const LIMIT: f32 = 100.;
				let min_avg = (500 >> event.older_times.len().min(16)) as f32;
				let is_outlier = new_avg > (old_avg * LIMIT).max(min_avg);
				if is_outlier {
					if let Some(outlier_schedule) = &mut event.outlier_schedule {
						outlier_schedule.run(world);
					} else {
						// ??? Ignore, crash, warning, etc.
						// ??? If ignored, clear the recent average?
						println!(
							"event {:?} is repeating >{}x more than normal at time {:?}\n\
							old avg: {:?}/s\n\
							new avg: {:?}/s",
							key,
							LIMIT,
							duration,
							old_avg,
							new_avg,
						);
						event.begin_schedule.run(world);
					}
				}
				
				 // Call Event:
				else {
					event.begin_schedule.run(world);
				}
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

/// Unique identifier for a case of prediction.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
struct PredId(u128);

#[derive(Default)]
pub struct PredHasher {
	bytes: [u8; std::mem::size_of::<PredId>()],
	index: usize,
}

impl PredHasher {
	fn write(&mut self, bytes: &[u8]) {
		let next_index = self.index + bytes.len();
		assert!(next_index <= self.bytes.len(), "overflowed maximum id size");
		self.bytes[self.index..next_index]
			.copy_from_slice(bytes);
		self.index = next_index;
	}
	fn finish(&self) -> PredId {
		PredId(u128::from_ne_bytes(self.bytes))
	}
}

/// A case of prediction, hashable into a unique identifier.
pub trait PredHash: Copy + Clone {
	fn pred_hash(self, state: &mut PredHasher);
}

impl PredHash for () {
	fn pred_hash(self, state: &mut PredHasher) {
		state.write(&[]);
	}
}

impl PredHash for Entity {
	fn pred_hash(self, state: &mut PredHasher) {
		self.to_bits().pred_hash(state);
	}
}

impl<const SIZE: usize> PredHash for [Entity; SIZE] {
	fn pred_hash(mut self, state: &mut PredHasher) {
		self.sort_unstable();
		for ent in self {
			ent.pred_hash(state);
		}
	}
}

macro_rules! impl_pred_case_for_ints {
	($($int:ty),+) => {
		$(
			impl PredHash for $int {
				fn pred_hash(self, state: &mut PredHasher) {
					state.write(&self.to_ne_bytes());
				}
			}
		)+
	};
}
impl_pred_case_for_ints!(
	u8, u16, u32, u64, u128, usize,
	i8, i16, i32, i64, i128, isize
);

impl<A: PredHash> PredHash for (A,) {
	fn pred_hash(self, state: &mut PredHasher) {
		self.0.pred_hash(state);
	}
}

impl<A: PredHash, B: PredHash> PredHash for (A, B) {
	fn pred_hash(self, state: &mut PredHasher) {
		self.0.pred_hash(state);
		self.1.pred_hash(state);
	}
}

impl<A: PredHash, B: PredHash, C: PredHash> PredHash for (A, B, C) {
	fn pred_hash(self, state: &mut PredHasher) {
		self.0.pred_hash(state);
		self.1.pred_hash(state);
		self.2.pred_hash(state);
	}
}

impl<A: PredHash, B: PredHash, C: PredHash, D: PredHash> PredHash for (A, B, C, D) {
	fn pred_hash(self, state: &mut PredHasher) {
		self.0.pred_hash(state);
		self.1.pred_hash(state);
		self.2.pred_hash(state);
		self.3.pred_hash(state);
	}
}

/// Collects predictions from "when" systems, for later compilation.
pub struct PredState<P: PredHash = ()>(Vec<(Box<dyn Iterator<Item = (Duration, Duration)> + Send + Sync>, P)>);

impl<P: PredHash> PredState<P> {
	pub fn set<I>(&mut self, case: P, times: TimeRanges<I>)
	where
		TimeRanges<I>: Iterator<Item = (Duration, Duration)> + Send + Sync + 'static
	{
		self.0.push((Box::new(times), case));
	}
}

struct ChimeEvent {
	times: Box<dyn Iterator<Item = (Duration, Duration)> + Send + Sync>,
	next_time: Option<Duration>,
	next_end_time: Option<Duration>,
	curr_time: Option<Duration>,
	prev_time: Option<Duration>,
	receivers: Vec<PredId>,
	begin_schedule: Schedule,
	end_schedule: Schedule,
	outlier_schedule: Option<Schedule>,
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
			receivers: Vec::new(),
			begin_schedule: Schedule::default(),
			end_schedule: Schedule::default(),
			outlier_schedule: None,
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

/// Event handler.
#[derive(Resource, Default)]
struct ChimeEventMap {
	/// All events, distinguished per prediction system and the system's cases.
	table: Vec<HashMap<PredId, ChimeEvent>>,
	
	/// Reverse time-to-events map for quickly rescheduling events.
	time_event_map: BTreeMap<Duration, Vec<ChimeEventKey>>,
}

type ChimeEventKey = (usize, PredId);

impl ChimeEventMap {
	fn first_time(&self) -> Option<Duration> {
		if let Some((&duration, _)) = self.time_event_map.first_key_value() {
			Some(duration)
		} else {
			None
		}
	}
	
	fn setup_id(&mut self) -> usize {
		self.table.push(Default::default());
		self.table.len() - 1
	}
	
	fn sched<Case: PredHash + Send + Sync + 'static>(
		&mut self,
		input: PredState<Case>,
		pred_time: Duration,
		event_id: usize,
		begin_sys: Option<&dyn ChimeEventSystem<In=Case, Out=()>>,
		end_sys: Option<&dyn ChimeEventSystem<In=Case, Out=()>>,
		outlier_sys: Option<&dyn ChimeEventSystem<In=Case, Out=()>>,
	) {
		// let n = input.0.len();
		// let a_time = Instant::now();
		let events = self.table.get_mut(event_id)
			.expect("id must be initialized with PredMap::setup_id");
		
		for (new_times, pred_case) in input.0 {
			// let a_time = Instant::now();
			let mut pred_state = PredHasher::default();
			pred_case.pred_hash(&mut pred_state);
			let pred_id = pred_state.finish();
			let key = (event_id, pred_id);
			let event = events.entry(key.1).or_default();
			event.times = new_times;
			// let b_time = Instant::now();
			
			 // Store Receiver:
			if !event.receivers.contains(&pred_id) {
				event.receivers.push(pred_id);
				// !!! https://bevyengine.org/news/bevy-0-13/#more-flexible-one-shot-systems
				if let Some(sys) = begin_sys {
					sys.add_to_schedule(&mut event.begin_schedule, pred_case);
				}
				if let Some(sys) = end_sys {
					sys.add_to_schedule(&mut event.end_schedule, pred_case);
				}
				if let Some(sys) = outlier_sys {
					if event.outlier_schedule.is_none() {
						event.outlier_schedule = Some(Schedule::default());
					}
					sys.add_to_schedule(
						event.outlier_schedule.as_mut().unwrap(),
						pred_case
					);
				}
			}
			// let c_time = Instant::now();
			
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
			// let d_time = Instant::now();
			if next_time != event.curr_time {
				 // Remove Old Prediction:
				if let Some(time) = event.curr_time {
					if let btree_map::Entry::Occupied(mut e)
						= self.time_event_map.entry(time)
					{
						let list = e.get_mut();
						let pos = list.iter()
							.position(|k| *k == key)
							.expect("this should always work");
						list.swap_remove(pos);
						if list.is_empty() {
							e.remove();
						}
					} else {
						unreachable!()
					}
				}
				
				 // Insert New Prediction:
				event.curr_time = next_time;
				if let Some(time) = next_time {
					let list = self.time_event_map.entry(time)
						.or_default();
					
					if is_active {
						list.push(key); // End events (run first)
					} else {
						list.insert(0, key); // Begin events (run last)
					}
				}
			}
			// let e_time = Instant::now();
			
			// println!(">>A {:?}", b_time.duration_since(a_time));
			// println!(">>B {:?}", c_time.duration_since(b_time));
			// println!(">>C {:?}", d_time.duration_since(c_time));
			// println!(">>D {:?}", e_time.duration_since(d_time));
		}
		// println!("  compile: {:?} // {:?}", Instant::now().duration_since(a_time), n);
	}
	
	fn pop(&mut self) -> ChimeEventKey {
		let mut entry = self.time_event_map.first_entry()
			.expect("this should always work");
		
		let time = *entry.key();
		let list = entry.get_mut();
		let key = list.pop()
			.expect("this should always work");
		
		if list.is_empty() {
			entry.remove();
		}
		
		let event = self.table
			.get_mut(key.0).expect("this should always work")
			.get_mut(&key.1).expect("this should always work");
		
		debug_assert_eq!(event.curr_time, Some(time));
		event.prev_time = Some(time);
		
		 // Queue Up Next Prediction:
		event.is_active = !event.is_active;
		event.curr_time = event.next_time();
		if let Some(t) = event.curr_time {
			self.time_event_map.entry(t)
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
		
		key
	}
}

/// Bevy schedule for re-predicting & scheduling events.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct ChimeSchedule;

/// Context for a `bevy::time::Time`.
#[derive(Default)]
pub struct Chime;