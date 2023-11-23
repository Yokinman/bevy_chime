// use bevy::{
// 	ecs::prelude::*,
// 	DefaultPlugins
// };

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::AtomicU64;
use std::time::Duration;
use bevy::ecs::query::Has;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::ecs::system::SystemId;
use bevy::prelude::*;
use bevy::utils::HashSet;

use chime::{flux, Flux, FluxVec, Moment, sum::Sum, Times, TimeUnit};

#[derive(PartialOrd, PartialEq)]
#[flux(Sum<f64, 2> = {val} + spd.per(TimeUnit::Secs))]
#[derive(Component, Debug)]
struct PosX {
	val: f64,
	spd: SpdX,
}

#[derive(PartialOrd, PartialEq)]
#[flux(Sum<f64, 1> = {val} + acc.per(TimeUnit::Secs))]
#[derive(Component, Debug)]
struct SpdX {
	val: f64,
	acc: AccX,
}

#[derive(PartialOrd, PartialEq)]
#[flux(Sum<f64, 0> = {val})]
#[derive(Component, Debug)]
struct AccX {
	val: f64,
}

#[derive(Component, Debug)]
struct Pos([<PosX as Moment>::Flux; 2]);

#[derive(Bundle, Debug)]
struct Dog {
	pos: Pos,
}

#[derive(Component, Debug)]
struct Gun;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct UniqueId(u64);

impl Default for UniqueId {
	fn default() -> Self {
		static ID: AtomicU64 = AtomicU64::new(0);
		Self(ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
		// !!! Ordering::Relaxed is probably fine, but I don't know for sure
	}
}


fn when_func_a(In(mut pred): In<PredCollector<Entity>>, query: Query<(&Pos, Entity), Changed<Pos>>) -> PredCollector<Entity> {
	for (pos, entity) in &query {
		let times: Times = pos.0[0].when_eq(&chime::Constant::from(100.))
			.chain(pos.0[0].when_eq(&chime::Constant::from(-100.)))
			.collect();
		
		println!("X: {:?}", times.clone().collect::<Vec<_>>());
		pred.add(times, entity);
	}
	pred
}

fn do_func_a(ent: Res<PredData>, time: Res<PredTime>, mut query: Query<(&mut Pos, Entity)>) {
	println!("!!! do some X");
	let ent = match ent.receiver {
		PredId::Entity(e) => e,
		_ => unreachable!(),
	};
	let time = time.0;
	
	let mut pos = (&mut query).into_iter()
		.find(|(_, entity)| entity == &ent)
		.map(|(pos, _)| pos)
		.unwrap();
	let mut pos_x = pos.0[0].at_mut(time);
	
	// let cool = ((30. + 5000. * (time.as_millis() as f64).cos().abs()).ceil() as i64).abs();
	let cool = pos_x.spd.val.abs();
	pos_x.spd.val = cool * -pos_x.spd.val.signum();
}

fn when_func_b(In(mut pred): In<PredCollector<Entity>>, query: Query<(&Pos, Entity), Changed<Pos>>) -> PredCollector<Entity> {
	for (pos, entity) in &query {
		let times: Times = pos.0[1].when_eq(&chime::Constant::from(100.))
			.chain(pos.0[1].when_eq(&chime::Constant::from(-100.)))
			.collect();
		
		println!("Y: {:?}", times.clone().collect::<Vec<_>>());
		pred.add(times, entity);
	}
	pred
}

fn do_func_b(ent: Res<PredData>, time: Res<PredTime>, mut query: Query<(&mut Pos, Entity)>) {
	println!("!!! do some Y");
	let ent = match ent.receiver {
		PredId::Entity(e) => e,
		_ => unreachable!(),
	};
	let time = time.0;
	
	let mut pos = (&mut query).into_iter()
		.find(|(_, entity)| entity == &ent)
		.map(|(pos, _)| pos)
		.unwrap();
	let mut pos_y = pos.0[1].at_mut(time);
	
	dbg!(pos_y.val.round(), pos_y.spd.val.abs());
	if pos_y.spd.val > -(2. * pos_y.spd.acc.val.abs()).sqrt() {
		pos_y.spd.val = 0.;
		pos_y.spd.acc.val = 0.;
	} else {
		pos_y.spd.val *= -1.;
	}
	drop(pos_y);
	
	// PosXFlux {
	// val: Constant { value: -100.00000024676038, time: 11.419693619s },
	// spd: SpdXFlux { val: Constant { value: 454.4060450826906, time: 11.419693619s }, acc: AccXFlux { val: Constant { value: -1000.0, time: 11.419693619s } } } }
	// :: Poly(Sum(-100.00000024676038, [454.4060450826906, -500.0]))  
	
	// let mut pos_x = pos.0[0].at_mut(time);
	// pos_x.spd.val = pos_x.spd.val;
}

fn when_func_c(In(mut pred): In<PredCollector<(Entity, Entity)>>, query: Query<(&Pos, Entity), Changed<Pos>>, b_query: Query<(&Pos, Entity)>) -> PredCollector<(Entity, Entity)> {
	for (pos, entity) in &query {
		for (b_pos, b_entity) in &b_query { // Actually, just iterate over the remainder of query A, cause it's 2-way
			let times: Times = pos.0.as_slice()
				.when_dis_eq(b_pos.0.as_slice(), &chime::Constant::from(16. + 16.));
			println!("DIS: {:?}", times.clone().collect::<Vec<_>>());
			pred.add(times, (entity.min(b_entity), b_entity.max(entity)));
		}
	}
	pred
}

fn do_func_c(ent: Res<PredData>, time: Res<PredTime>, mut query: Query<(&mut Pos, Entity)>) {
	println!("!!! do some DIS");
	let (ent, b_ent) = match ent.receiver {
		PredId::Entity2(e1, e2) => (e1, e2),
		_ => unreachable!(),
	};
	let time = time.0;
	
	let mut pos = None;
	let mut b_pos = None;
	
	for (p, entity) in &mut query {
		if entity == ent {
			pos = Some(p);
		} else if entity == b_ent {
			b_pos = Some(p);
		}
	}
	
	let mut pos = pos.unwrap();
	let mut pos = pos.0.at_mut(time);
	let b_pos = b_pos.unwrap();
	let b_pos = b_pos.0.at(time);
	// println!("{:?}, {:?}", pos, b_pos);
	
	let x = pos[0].val - b_pos[0].val;
	let y = pos[1].val - b_pos[1].val;
	let dir = (y as f64).atan2(x as f64);
	let spd = ((pos[0].spd.val as f64)*(pos[0].spd.val as f64) + (pos[1].spd.val as f64)*(pos[1].spd.val as f64)).sqrt();
	pos[0].spd.val = (spd * dir.cos()) as f64;
	pos[1].spd.val = (spd * dir.sin()) as f64;
}

fn world_add_when<Case, WhenMarker, WhenSys, DoMarker, DoSys>(
	world: &mut World,
	when_system: WhenSys,
	do_system: DoSys,
)
where
	Case: PredCase + 'static,
	WhenSys: IntoSystem<PredCollector<Case>, PredCollector<Case>, WhenMarker> + 'static,
	DoSys: IntoSystem<(), (), DoMarker> + 'static,
{
	let do_id = world.register_system(do_system);
	let unique_id = UniqueId::default();
	
	let input = || -> PredCollector<Case> {
		PredCollector(Vec::new())
	};
	
	let compile = move |In(input): In<PredCollector<Case>>, mut pred: ResMut<PredMap>, pred_time: Res<PredTime>| {
		for (times, case) in input.0 {
			let pred_key = (unique_id, case.into_id());
			pred.sched(pred_key, pred_time.0, times, do_id);
		}
	};
	
	world.resource_mut::<Schedules>()
		.get_mut(PredSchedule).unwrap()
		.add_systems(input.pipe(when_system).pipe(compile));
}

fn setup(world: &mut World) {
	world.insert_resource(PredTime(Duration::ZERO));
	world.insert_resource(PredMap::default());
	world.insert_resource(PredData {
		receiver: PredId::None,
		can_repeat: false,
		is_repeating: false,
	});
	
	let schedule = Schedule::new(PredSchedule);
	world.add_schedule(schedule);
	world_add_when(world, when_func_a, do_func_a);
	world_add_when(world, when_func_b, do_func_b);
	world_add_when(world, when_func_c, do_func_c);
	
    world.spawn(Camera2dBundle::default());
	world.spawn((
		Dog {
			pos: Pos([
				PosX { val: 0., spd: SpdX { val: 100., acc: AccX { val: 0. } } },
				PosX { val: 0., spd: SpdX { val: 0.,  acc: AccX { val: 0. } } }
			].to_flux(Duration::ZERO)),
		},
		SpriteBundle {
			transform: Transform::from_xyz(100.0, 5.0, 20.0),
			texture: world.resource::<AssetServer>().load("textures/air_unit.png"),
			..default()
		},
		Gun,
	));
	world.spawn((
		Dog {
			pos: Pos([
				PosX { val: 0., spd: SpdX { val: 60., acc: AccX { val: 0. } } },
				PosX { val: 0., spd: SpdX { val: 0., acc: AccX { val: 0. } } }
			].to_flux(Duration::ZERO)),
		},
		SpriteBundle::default(),
	));
}

fn update(world: &mut World) {
	world.run_schedule(PredSchedule);
	
	/* !!! Issues:
	   
		[ ] Squeezing:
		
		Pass in a resource to events that can be used to define whether the
		current event can repeat and knows whether it has already. Then, events
		can use this to conveniently handle to sub-nanosecond loops.
		
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
		
		[ ] Repetition
		
		An event might execute more than once if its prediction is slightly
		early or late due to imprecision, leading to the event producing another
		prediction (e.g. a ball bouncing off a wall; if a prediction is slightly
		late, it bounces once and then again off the opposite side of the wall).
		
		This has been partially fixed by rounding predicted times toward the
		current time, but not fully. For extreme cases, one potential solution
		might be to round predictions to a range of nanoseconds, but this can
		only go so far. I'm not sure how extreme the error can get or what it
		scales by.
		
		[ ] Rapid Repetition
		
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
	
	let time = world.resource::<Time>().elapsed();
	
	while let Some(&(duration, system, key)) = world.resource::<PredMap>().time_stack.last() {
		if time >= duration {
			println!();
			// println!("first: {:?}", world.resource::<PredMap>().time_stack);
			// println!("also: {:?}", world.resource::<PredMap>().time_table);
			world.resource_mut::<PredTime>().0 = duration;
			world.resource_mut::<PredMap>().pop();
			
			println!("B: {:?}, {:?}", time, duration);
			
			let receivers = world.resource::<PredMap>().time_table.get(&key)
				.unwrap().receivers.clone();
			for receiver in receivers {
				world.resource_mut::<PredData>().receiver = receiver;
				world.run_system(system);
			}
			
			world.run_schedule(PredSchedule);
		} else {
			break
		}
	}
	
	world.resource_mut::<PredTime>().0 = time;
}

fn debug_draw(mut draw: Gizmos, time: Res<Time>, pred_time: Res<PredTime>, query: Query<(&Pos, Has<Gun>)>) {
	let s = time.elapsed_seconds();
	for (pos, has_gun) in &query {
		let x = pos.0[0].at(pred_time.0);
		let y = pos.0[1].at(pred_time.0);
		let pos = Vec2::new(x.val as f32, y.val as f32);
		draw.circle_2d(pos, 16.0, Color::BLUE);
		if has_gun {
			draw.line_2d(
				pos,
				pos + Vec2::new(16.* s.cos(), 16.* s.sin()),
				Color::GREEN
			);
		}
	}
	// for (pos, _) in &query {
	// 	let x = pos.0[0].at(pred_time.0);
	// 	let y = pos.0[1].at(pred_time.0);
	// 	let pos = Vec2::new(x.val as f32, y.val as f32);
	// 	draw.line_2d(Vec2::new(pos.x, -100.0), Vec2::new(pos.x, 100.0), Color::BEIGE);
	// }
	draw.rect_2d(Vec2::ZERO, 0.0, Vec2::ONE*200.0, Color::LIME_GREEN);
}

/// Specific data from the "when" prediction fed into the action.
#[derive(Resource, Debug)]
struct PredData {
	receiver: PredId,
	can_repeat: bool,
	is_repeating: bool,
}

/// What uniquely identifies a case of prediction.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PredId {
	None,
	Entity(Entity),
	Entity2(Entity, Entity),
}

impl PredId {
	pub fn simplify(self) -> Self {
		match self {
			PredId::Entity2(a, b) if a > b => {
				PredId::Entity2(b, a)
			},
			other => other
		}
	}
}

/// A case of prediction, like what it's based on.
trait PredCase {
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

impl PredCase for (Entity, Entity) {
	fn into_id(self) -> PredId {
		PredId::Entity2(self.0, self.1)
	}
}

/// Collects predictions from "when" systems, for later compilation.
struct PredCollector<Case: PredCase = ()>(Vec<(Times, Case)>);

impl<Case: PredCase> PredCollector<Case> {
	fn add(&mut self, times: Times, case: Case) {
		self.0.push((times, case));
	}
}

#[derive(Default)]
struct PredCaseData {
	time_index: usize,
	time_list: Vec<Duration>,
	receivers: HashSet<PredId>,
	// can_repeat: bool,
	last_time: Duration,
	// repeat_count: usize,
}

/// ...
#[derive(Resource, Default)]
pub struct PredMap {
	time_stack: Vec<(Duration, SystemId, PredKey)>,
	time_table: HashMap<PredKey, PredCaseData>,
	// repeated_case: Option<PredKey>,
}

type PredKey = (UniqueId, PredId);

impl PredMap {
	pub fn sched(&mut self, key: PredKey, pred_time: Duration, mut times: Times, system: SystemId) {
		//               v' v
		// [ 1, 3, 8, 8, 9, 9, 9,  12 ]
		// [ 0, 3, 5, 8, 8, 9, 12, 15 ]
		// !!! Make sure `times` is ordered.
		let mut v = times.into_iter().collect::<Vec<Duration>>();
		v.sort_unstable();
		println!("NEW: {:?} at {:?} with {:?}", v, pred_time, key);
		times = v.into_iter().collect();
		
		let (unique_id, pred_id) = key;
		let key = (unique_id, pred_id.simplify());
		
		 // Store in Table:
		let PredCaseData {
			time_index: curr_index,
			time_list: curr_times,
			receivers: curr_cases,
			..
		} = self.time_table.entry(key).or_default();
		
		curr_cases.insert(pred_id);
		
		println!("OLD: {:?}, {:?}", curr_times, curr_index);
		let prev_index = std::mem::take(curr_index);
		let mut old_times = std::mem::take(curr_times).into_iter();
		let mut old_time = old_times.next();
		let mut old_index = 0;
		let mut time = times.next();
		let mut index = 0;
		'a: loop {
			let o = match (time.as_ref(), old_time.as_ref()) {
				(Some(a), Some(b)) => {
					a.cmp(b) // !!! Can we make the prediction more precise instead
					// a.as_secs().cmp(&b.as_secs())
					// 	.then((a.as_micros()).cmp(&(b.as_micros())))
				},
				(Some(_), _) => Ordering::Less,
				(_, Some(_)) => Ordering::Greater,
				_ => break 'a,
			};
			match o {
				Ordering::Less => {
					 // Sort Into Main Queue:
					if time.unwrap() >= pred_time {
						let insert_index = self.time_stack.partition_point(
							|(t, ..)| t > &time.unwrap()
						);
						self.time_stack.insert(insert_index, (time.unwrap(), system, key));
					} else {
						*curr_index += 1;
					}
					
					 // Next New Time:
					curr_times.push(time.unwrap());
					time = times.next();
					index += 1;
				},
				Ordering::Greater => {
					 // Remove First Instance From Main Queue:
					if old_index >= prev_index {   
						let mut stack_index = self.time_stack.partition_point(
							|(t, ..)| t >= &old_time.unwrap()
						);
						if stack_index == 0 {
							unreachable!()
						}
						// println!("A {:?} :: {:?}", self.time_stack, old_time);
						loop {
							stack_index -= 1;
							let (.., k) = self.time_stack[stack_index];
							if k == key {
								self.time_stack.remove(stack_index);
								break
							}
						}
						// println!("B {:?} :: {:?}", self.time_stack, old_time);
					}
					
					 // Next Old Time:
					old_time = old_times.next();
					old_index += 1;
				},
				Ordering::Equal => {
					
					 // Preserve Old Predictions:
					if old_index < prev_index {
						*curr_index += 1;
					}
					
					 // Next Times:
					curr_times.push(time.unwrap());
					time = times.next();
					index += 1;
					old_time = old_times.next();
					old_index += 1;
				},
			}
		}
		println!("FIN: {:?}, {:?}", curr_times, curr_index);
	}
	
	pub fn pop(&mut self) {
		let (time, _, key) = self.time_stack.pop().unwrap();
		let PredCaseData {
			time_index,
			last_time,
			..
		} = self.time_table.get_mut(&key).unwrap();
		*time_index += 1;
		*last_time = time;
		// if *index == times.len() {
		// 	println!("ding! gone! LOL {:?} {:?}", times, index);
		// 	self.time_table.remove(&key); // !!! Do this after the main loop
		// }
	}
}

#[derive(Resource)]
struct PredTime(Duration);

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct PredSchedule;

/// ...
struct Game;
impl Plugin for Game {
	fn build(&self, app: &mut App) {
		app.add_plugins(DefaultPlugins);
		app.add_systems(Startup, setup);
		app.add_systems(Update, update);
		app.add_systems(Update, debug_draw);
	}
}

fn main() {
	use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
	App::new()
		.add_plugins(Game)
        // .add_plugins(LogDiagnosticsPlugin::default())
        // .add_plugins(FrameTimeDiagnosticsPlugin::default())
		.run();
}
