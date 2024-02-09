// use bevy::{
// 	ecs::prelude::*,
// 	DefaultPlugins
// };

use std::cmp::Reverse;
use std::collections::{BinaryHeap, btree_map, BTreeMap, HashMap};
use std::hash::Hash;

use bevy::app::{App, Plugin, Startup, Update};
use bevy::ecs::entity::Entity;
use bevy::ecs::schedule::{Schedule, Schedules, ScheduleLabel};
use bevy::ecs::system::{In, IntoSystem, Res, ResMut, Resource, ReadOnlySystem, System};
use bevy::ecs::world::{Mut, World};

use std::time::{Duration, Instant};
use bevy::time::Time;
use chime::time::TimeRanges;

use bevy::input::{Input, keyboard::KeyCode};

pub fn world_add_chime_system<Case, WhenMarker, WhenSys, BeginMarker, BeginSys, EndMarker, EndSys, OutlierMarker, OutlierSys>(
	world: &mut World,
	when_system: WhenSys,
	begin_system: BeginSys,
	end_system: EndSys,
	outlier_system: OutlierSys,
)
where
	Case: PredHash + Send + Sync + 'static,
	WhenSys: IntoSystem<PredCollector<Case>, PredCollector<Case>, WhenMarker> + 'static,
	WhenSys::System: ReadOnlySystem,
	BeginSys: IntoSystem<Case, (), BeginMarker> + Copy + Send + Sync + 'static,
	EndSys: IntoSystem<Case, (), EndMarker> + Copy + Send + Sync + 'static,
	OutlierSys: IntoSystem<Case, (), OutlierMarker> + Copy + Send + Sync + 'static,
{
	let id = world.resource_mut::<PredMap>().setup_id();
	
	let input = || -> PredCollector<Case> {
		PredCollector(Vec::new())
	};
	
	let compile = move |In(input): In<PredCollector<Case>>, mut pred: ResMut<PredMap>, pred_time: Res<Time>| {
		pred.sched(input, pred_time.elapsed(), id, begin_system, end_system, outlier_system);
	};
	
	world.resource_mut::<Schedules>()
		.get_mut(ChimeSchedule).unwrap()
		.add_systems(input.pipe(when_system).pipe(compile));
}

fn setup(world: &mut World) {
	world.insert_resource(Time::<Chime>::default());
	world.insert_resource(PredMap::default());
	world.add_schedule(Schedule::new(ChimeSchedule));
}

const RECENT_TIME: Duration = Duration::from_millis(500);
const OLDER_TIME: Duration = Duration::from_secs(10);

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
	
	let can_can_print = world.resource::<Input<KeyCode>>().pressed(KeyCode::Space);
	let mut can_print = can_can_print;
	
	pred_schedule.run(world);
	
	while let Some(duration) = world.resource::<PredMap>().first_time() {
		if time >= duration {
			world.resource_mut::<Time>().advance_to(duration);
			
			if can_print {
				can_print = false;
				println!("Time: {:?}", duration);
			}
			
			let a_time = Instant::now();
			if world.resource_scope(|world, mut pred: Mut<PredMap>| -> bool {
				let key = pred.pop();
				let case = pred.time_table
					.get_mut(key.0).unwrap()
					.get_mut(&key.1).unwrap();
				
				if !case.is_active {
					case.end_schedule.run(world);
					return false
				}
				// let last_time = std::mem::replace(&mut case.last_time, Some(duration));
				
				 // Ignore Rapidly Repeating Events:
				let new_avg = (case.recent_times.len() as f32) / RECENT_TIME.as_secs_f32();
				let old_avg = (case.older_times.len() as f32) / OLDER_TIME.as_secs_f32();
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
				let min_avg = (500 >> case.older_times.len().min(16)) as f32;
				let is_outlier = new_avg > (old_avg * LIMIT).max(min_avg);
				if is_outlier {
					if let Some(outlier_schedule) = &mut case.outlier_schedule {
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
						return true // Don't reschedule
					}
				}
				
				 // Call Event:
				else {
					case.schedule.run(world);
				}
				
				false
			}) {
				continue
			}
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
pub struct PredId(u128);

#[derive(Default)]
pub struct PredCaseHasher {
	bytes: [u8; std::mem::size_of::<PredId>()],
	index: usize,
}

impl PredCaseHasher {
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

/// A case of prediction, like what it's based on.
pub trait PredHash: Copy + Clone {
	fn pred_hash(self, state: &mut PredCaseHasher);
}

impl PredHash for () {
	fn pred_hash(self, state: &mut PredCaseHasher) {
		state.write(&[]);
	}
}

impl PredHash for Entity {
	fn pred_hash(self, state: &mut PredCaseHasher) {
		self.to_bits().pred_hash(state);
	}
}

impl<const SIZE: usize> PredHash for [Entity; SIZE] {
	fn pred_hash(mut self, state: &mut PredCaseHasher) {
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
				fn pred_hash(self, state: &mut PredCaseHasher) {
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
	fn pred_hash(mut self, state: &mut PredCaseHasher) {
		self.0.pred_hash(state);
	}
}

impl<A: PredHash, B: PredHash> PredHash for (A, B) {
	fn pred_hash(mut self, state: &mut PredCaseHasher) {
		self.0.pred_hash(state);
		self.1.pred_hash(state);
	}
}

impl<A: PredHash, B: PredHash, C: PredHash> PredHash for (A, B, C) {
	fn pred_hash(mut self, state: &mut PredCaseHasher) {
		self.0.pred_hash(state);
		self.1.pred_hash(state);
		self.2.pred_hash(state);
	}
}

impl<A: PredHash, B: PredHash, C: PredHash, D: PredHash> PredHash for (A, B, C, D) {
	fn pred_hash(mut self, state: &mut PredCaseHasher) {
		self.0.pred_hash(state);
		self.1.pred_hash(state);
		self.2.pred_hash(state);
		self.3.pred_hash(state);
	}
}

/// Collects predictions from "when" systems, for later compilation.
pub struct PredCollector<Case: PredHash = ()>(Vec<(TimeRanges, Case)>);

impl<Case: PredHash> PredCollector<Case> {
	pub fn add(&mut self, times: TimeRanges, case: Case) {
		self.0.push((times, case));
	}
}

#[derive(Default)]
struct PredCaseData {
	times: TimeRanges,
	next_time: Option<Duration>,
	next_end_time: Option<Duration>,
	last_time: Option<Duration>,
	prev_time: Option<Duration>,
	receivers: Vec<PredId>,
	schedule: Schedule,
	end_schedule: Schedule,
	outlier_schedule: Option<Schedule>,
	recent_times: BinaryHeap<Reverse<Duration>>,
	older_times: BinaryHeap<Reverse<Duration>>,
	is_active: bool,
}

impl PredCaseData {
	fn next_time(&mut self) -> Option<Duration> {
		let next_time = if self.is_active {
			&mut self.next_end_time
		} else {
			&mut self.next_time
		};
		if next_time.is_some() {
			return std::mem::take(next_time)
		}
		
		 // :
		if let Some((a, b)) = self.times.next() {
			self.next_time = Some(a);
			self.next_end_time = Some(b);
			self.next_time()
		} else {
			None
		}
	}
}

pub fn temp_default_outlier<T: PredHash>(_: In<T>) {}

/// Event handler.
#[derive(Resource, Default)]
struct PredMap {
	time_stack: BTreeMap<Duration, Vec<PredKey>>,
	time_table: Vec<HashMap<PredId, PredCaseData>>,
}

type PredKey = (usize, PredId);

impl PredMap {
	fn first_time(&self) -> Option<Duration> {
		if let Some((&duration, _)) = self.time_stack.first_key_value() {
			Some(duration)
		} else {
			None
		}
	}
	
	fn setup_id(&mut self) -> usize {
		self.time_table.push(Default::default());
		self.time_table.len() - 1
	}
	
	fn sched<Case, BeginMarker, BeginSys, EndMarker, EndSys, OutlierMarker, OutlierSys>(
		&mut self,
		input: PredCollector<Case>,
		pred_time: Duration,
		system_id: usize,
		system: BeginSys,
		end_system: EndSys,
		outlier_system: OutlierSys,
	)
	where
		Case: PredHash + Send + Sync + 'static,
		BeginSys: IntoSystem<Case, (), BeginMarker> + Copy + Sync + 'static,
		EndSys: IntoSystem<Case, (), EndMarker> + Copy + Sync + 'static,
		OutlierSys: IntoSystem<Case, (), OutlierMarker> + Copy + Sync + 'static,
	{
		// let n = input.0.len();
		// let a_time = Instant::now();
		let table = self.time_table.get_mut(system_id)
			.expect("id must be initialized with PredMap::setup_id");
		
		for (new_times, pred_case) in input.0 {
			// let a_time = Instant::now();
			let mut pred_state = PredCaseHasher::default();
			pred_case.pred_hash(&mut pred_state);
			let pred_id = pred_state.finish();
			let key = (system_id, pred_id);
			let case = table.entry(key.1).or_default();
			case.times = new_times;
			// let b_time = Instant::now();
			
			 // Store Receiver:
			if !case.receivers.contains(&pred_id) {
				case.receivers.push(pred_id);
				let input = move || -> Case { pred_case };
				case.schedule.add_systems(input.pipe(system));
				case.end_schedule.add_systems(input.pipe(end_system));
				if IntoSystem::into_system(outlier_system).name()
					!= IntoSystem::into_system(temp_default_outlier::<Case>).name()
				{
					if case.outlier_schedule.is_none() {
						case.outlier_schedule = Some(Schedule::default());
					}
					case.outlier_schedule.as_mut().unwrap().add_systems(input.pipe(outlier_system));
				}
			}
			// let c_time = Instant::now();
			
			 // Fetch Next Time:
			case.next_time = None;
			let prev_time = std::mem::take(&mut case.prev_time);
			let mut is_active = std::mem::take(&mut case.is_active);
			let mut next_time = None;
			loop {
				next_time = case.next_time();
				if let Some(t) = next_time {
					if t > pred_time || (t == pred_time && !is_active && prev_time != Some(t)) {
						break
					}
					case.prev_time = Some(t);
					case.is_active = !case.is_active;
				} else {
					break
				}
			}
			if case.is_active != is_active {
				case.is_active = is_active;
				if is_active {
					case.next_time = next_time;
				} else {
					case.next_end_time = next_time;
				}
				next_time = Some(pred_time);
			}
			
			 // Update Prediction:
			// let d_time = Instant::now();
			if next_time != case.last_time {
				 // Remove Old Prediction:
				if let Some(time) = case.last_time {
					if let btree_map::Entry::Occupied(mut e)
						= self.time_stack.entry(time)
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
				case.last_time = next_time;
				if let Some(time) = next_time {
					let mut list = self.time_stack.entry(time)
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
	
	fn pop(&mut self) -> PredKey {
		let mut entry = self.time_stack.first_entry()
			.expect("this should always work");
		
		let time = *entry.key();
		let list = entry.get_mut();
		let key = list.pop()
			.expect("this should always work");
		
		if list.is_empty() {
			entry.remove();
		}
		
		let case = self.time_table
			.get_mut(key.0).expect("this should always work")
			.get_mut(&key.1).expect("this should always work");
		
		debug_assert_eq!(case.last_time, Some(time));
		case.prev_time = Some(time);
		
		 // Queue Up Next Prediction:
		case.is_active = !case.is_active;
		case.last_time = case.next_time();
		if let Some(t) = case.last_time {
			self.time_stack.entry(t)
				.or_default()
				.push(key);
		}
		
		 // Overall vs Recent Average:
		while let Some(Reverse(t)) = case.older_times.peek() {
			if time < *t {
				break
			}
			case.older_times.pop();
		}
		while let Some(Reverse(t)) = case.recent_times.peek() {
			if time < *t {
				break
			}
			case.older_times.push(Reverse(*t + OLDER_TIME));
			case.recent_times.pop();
		}
		case.recent_times.push(Reverse(time + RECENT_TIME));
		
		key
	}
}

/// Bevy schedule for re-predicting & scheduling events.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct ChimeSchedule;

/// Context for a `bevy::time::Time`.
#[derive(Default)]
pub struct Chime;

/// ...
pub struct ChimePlugin;

impl Plugin for ChimePlugin {
	fn build(&self, app: &mut App) {
		app.add_systems(Startup, setup);
		app.add_systems(Update, update);
	}
}