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
use bevy::ecs::system::{In, IntoSystem, Res, ResMut, Resource, ReadOnlySystem};
use bevy::ecs::world::{Mut, World};

use std::time::{Duration, Instant};
use bevy::time::Time;
use chime::time::Times;

use bevy::input::{Input, keyboard::KeyCode};

pub fn world_add_chime_system<Case, WhenMarker, WhenSys, DoMarker, DoSys>(
	world: &mut World,
	when_system: WhenSys,
	do_system: DoSys,
)
where
	Case: PredCase + Send + Sync + 'static,
	WhenSys: IntoSystem<PredCollector<Case>, PredCollector<Case>, WhenMarker> + 'static,
	WhenSys::System: ReadOnlySystem,
	DoSys: IntoSystem<Case, (), DoMarker> + Copy + Send + Sync + 'static,
{
	let id = world.resource_mut::<PredMap>().setup_id();
	
	let input = || -> PredCollector<Case> {
		PredCollector(Vec::new())
	};
	
	let compile = move |In(input): In<PredCollector<Case>>, mut pred: ResMut<PredMap>, pred_time: Res<Time>| {
		pred.sched(input, pred_time.elapsed(), id, do_system);
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
	/* !!! Issues:
	   
		[~] Squeezing:
		
		Pass in a resource to events that can be used to define whether the
		current event can repeat and knows whether it has already. Then, events
		can use this to conveniently handle sub-nanosecond loops.
		
		A ball pushed into a wall by another ball could either state:
		
		A. If I'm repeating ball collision, push the other ball away instead of
		   bouncing off of it.
		B. If I'm repeating wall collision, freeze my velocity.
		
		Events that occurred before a repeatable event should also be
		repeatable, even if they aren't independently so (A/B/A vs B/A/B/A).
		
		The resource could also store a slice of the events that have run
		between repetitions for the event to react to. It could also store a
		count, but the event's system could store that locally if needed.
		
		To avoid infinite loops, the "repeatable" flag of an event is false by
		default, and should be reset just before the event is repeated.
		
		There's kind of two contexts - one in which an event invalidates itself,
		and one in which it doesn't. If an event invalidates itself, it can
		still infinite loop because there's a chance that it might break out of
		that kind of loop. It's equivalent to any other events that occur too
		often. However, if an event doesn't invalidate itself, then it won't
		reschedule itself and can be "squeezed".
		
		I'm unsure how to deal with events that don't invalidate themselves:
		
		- Right now events that reschedule themselves at the current time are
		  ignored, but this means that if a value is squeezed into the event
		  it'll just get ignored with no way to respond to it.
		- It shouldn't infinite loop, because the event is supposed to be a
		  discrete response to the value becoming the way that it is.
		- I could make it so if an event happens in between the squeezing that
		  modifies the value, then the event can be rescheduled. However, what
		  about a case like a ball bouncing off a wall with a bounce pad on it.
		  The bounce pad might just increase the ball's speed after it bounces
		  off the wall, but that change would make the bounce event run again.
		- It could be something specific where if an event flips the rate of its
		  predicate value, and then another event flips that rate, the original
		  event can be rescheduled. It just seems so specific to a bounce case.
		  • Case 1: When an object passes a boundary line, its friction toggles
		    between two different states (faster vs slower zones). If the object
		    is on the boundary line, and then gets bounced between two objects
		    that instantly disappear (all staying on the line), I don't think it
		    should repeat the boundary line toggle at all. Especially if the
		    order of events is unreliable.
		  • Case 2: When an object touches a boundary line, it flips its speed
		    so that it bounces back. If it bounces back and instantly rebounds
		    off of another object, it should rerun that initial bounce event.
		  • Case 3: When an object touches a boundary line, it flips its speed
		    so that it bounces back. On the line is also a "bounce pad" object
		    that increases its acceleration. If it bounces and then accelerates
		    then the initial event shouldn't run again.
		  
		It might make sense to track how often each event runs on average, and
		then compare that to a recent average. If the recent average is like
		1000x more often, then something is probably wrong. Maybe ignore periods
		of time in which the event is "inactive", when nothing is scheduled.
		
		[ ] Rounding:
		
		Using an integer type will cause "untouched" values to round when an
		overall structure is modified. This can cause repetitive events to be
		scheduled since the object may effectively "teleport" and collide with
		something over and over again.
		
		For example, there seems to be a "chain-link" issue where two balls
		moving away from each other slightly round into each other and get
		stuck. They don't infinite loop, but they get as close as they can.
		
		A. Remove float-integer isomorphisms since they aren't truly isomorphic.
		B. Make the integer types isomorphic by preserving in-betweens somehow.
		   It seems like you might as well just use f64 at that point, though.
		C. Heavily warn or make it obvious that using `at_mut` may modify values
		   even if they aren't touched. Somehow.
		D. Somehow register when a value is modified so that information isn't
		   lost unless an actual modification occurs.
		   - Wrap the moment's types (sounds annoying and unhygienic).
		   - Only update the flux's value if the moment's value was changed.
		     So `Moment.value = rnd(2.5) = 2, set to 2, Flux.value remains 2.5`.
		     Alternatively, maybe count changes as "offsets" instead.
		     So `Moment.value = rnd(2.5) = 2, set to 4, Flux.value becomes 4.5`.
		     
		I'm between A and D. The main inconvenience with D is that I'd have to
		handle it manually in the `Flux::set_moment` method - I think. I just
		wonder if there's a slightly better way to handle it.
		
		Honestly, option B might work best with a special deref wrapper that
		just preserves the fractional part of the original value. Stays simple.
		
		[~] Repetition
		
		An event might execute more than once if its prediction is slightly
		early or late due to imprecision, leading to the event producing another
		prediction (e.g. a ball bouncing off a wall; if a prediction is slightly
		late, it bounces once and then again off the opposite side of the wall).
		
		This has been partially fixed by rounding predicted times toward the
		current time, but not fully. For extreme cases, one potential solution
		might be to round predictions to a range of nanoseconds, but this can
		only go so far. I'm not sure how extreme the error can get or what it
		scales by.
		
		The precision can probably be improved a bit by optimizing the precision
		breakdown `LIMIT` constant, or something related to it. I think it does
		improve precision, but only when the normal calculation is sucky.
		
		[~] Rapid Repetition
		
		Values that are associated with events that occur very often can be
		extremely slow. For example, an object bouncing off of the floor with
		a large amount of gravity and a very low vertical position & speed. The
		bounce can effectively occur hundreds of thousands of times per second
		under the right conditions.
		
		A. Could force the events to handle it individually; like the bounce
		   event could cancel vertical acceleration if the speed is less than a
		   certain amount. Still, it seems an unintuitive thing to handle.
		B. Could there be some kind of limit on an event happening multiple
		   times within a certain timespan? Could be, but it would certainly
		   also seem unintuitive as to why the event suddenly didn't execute as
		   expected.
		   
		I think combining this with the squeezing fix might be the best option,
		where how repetitious an event can be before the default behavior occurs
		(ignoring, crashing, warning, waiting - maybe based on build type).
	*/
	
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
				let is_outlier = new_avg > old_avg.max(1.) * LIMIT;
				if is_outlier {
					if case.is_repeating {
						// ??? Ignore, crash, warning, etc.
						// ??? If ignored, clear the recent average?
						println!(
							"event {:?} is repeating {}x more than normal at time {:?}\n\
							old avg: {:?}/s\n\
							new avg: {:?}/s",
							key,
							LIMIT,
							duration,
							old_avg,
							new_avg,
						);
						return true
					} else {
						case.is_repeating = true;
					}
				} else {
					case.is_repeating = false;
				}
				
				 // Call Event:
				case.schedule.run(world);
				
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
	if b_time.duration_since(a_time) > chime::time::MILLISEC {
		println!("lag at {time:?} ({num:?}): {:?}", b_time.duration_since(a_time));
		println!("  run: {:?}", tot_a);
		println!("  pred: {:?}", tot_b);
	}
}

/// What uniquely identifies a case of prediction.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PredId {
	None,
	Entity(Entity),
	Entity2([Entity; 2]),
}

impl PredId {
	pub fn simplify(self) -> Self {
		match self {
			PredId::Entity2([a, b]) if a > b => {
				PredId::Entity2([b, a])
			},
			other => other
		}
	}
}

/// A case of prediction, like what it's based on.
pub trait PredCase: Copy + Clone {
	fn into_id(self) -> PredId;
}

impl PredCase for () {
	fn into_id(self) -> PredId {
		PredId::None
	}
}

impl PredCase for Entity {
	fn into_id(self) -> PredId {
		PredId::Entity(self)
	}
}

impl PredCase for [Entity; 2] {
	fn into_id(self) -> PredId {
		PredId::Entity2(self)
	}
}

/// Collects predictions from "when" systems, for later compilation.
pub struct PredCollector<Case: PredCase = ()>(Vec<(Times, Case)>);

impl<Case: PredCase> PredCollector<Case> {
	pub fn add(&mut self, times: Times, case: Case) {
		self.0.push((times, case));
	}
}

#[derive(Default)]
struct PredCaseData {
	times: Times,
	next_time: Option<Duration>,
	last_time: Option<Duration>,
	receivers: Vec<PredId>,
	schedule: Schedule,
	recent_times: BinaryHeap<Reverse<Duration>>,
	older_times: BinaryHeap<Reverse<Duration>>,
	is_repeating: bool,
}

impl PredCaseData {
	fn next_time(&mut self, time: Duration) -> Option<Duration> {
		while let Some(t) = self.times.next() {
			if t >= time && Some(t) != self.last_time {
				self.next_time = Some(t); 
				return self.next_time
			}
		}
		self.next_time = None;
		None
	}
}

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
	
	fn sched<Case, DoMarker, DoSys>(
		&mut self,
		input: PredCollector<Case>,
		pred_time: Duration,
		system_id: usize,
		system: DoSys,
	)
	where
		Case: PredCase + Send + Sync + 'static,
		DoSys: IntoSystem<Case, (), DoMarker> + Copy + Sync + 'static,
	{
		// let n = input.0.len();
		// let a_time = Instant::now();
		let table = self.time_table.get_mut(system_id)
			.expect("id must be initialized with PredMap::setup_id");
		
		for (new_times, pred_case) in input.0 {
			// let a_time = Instant::now();
			let pred_id = pred_case.into_id();
			let key = (system_id, pred_id.simplify());
			let case = table.entry(key.1).or_default(); // !!! very slow
			case.times = new_times;
			// let b_time = Instant::now();
			
			 // Store Receiver:
			if !case.receivers.contains(&pred_id) {
				case.receivers.push(pred_id);
				let input = move || -> Case { pred_case };
				case.schedule.add_systems(input.pipe(system));
			}
			// let c_time = Instant::now();
			
			let prev_time = std::mem::take(&mut case.next_time);
			let next_time = case.next_time(pred_time); // !!! a lil slow
			// let d_time = Instant::now();
			if next_time != prev_time {
				 // Can Reschedule at Current Time:
				case.last_time = None;
				
				 // Remove Old Prediction:
				if let Some(time) = prev_time {
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
				if let Some(time) = next_time {
					self.time_stack.entry(time)
						.or_default()
						.push(key);
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
		
		debug_assert_eq!(case.next_time, Some(time));
		
		 // Queue Up Next Prediction:
		case.last_time = Some(time);
		if let Some(t) = case.next_time(time) {
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