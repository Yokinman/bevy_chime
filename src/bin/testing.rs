use bevy_chime::*;

use bevy::prelude::*;
use bevy::window::PresentMode;

use chime::{flux, Flux, FluxVec, Moment, sum::Sum, time};
use chime::kind::{WhenDisEq};

use std::time::{Duration, Instant};

use impl_op::impl_op;

#[derive(PartialOrd, PartialEq)]
#[flux(Sum<f64, 2> = {val} + spd.per(time::SEC))]
#[derive(Component, Debug)]
struct PosX {
	val: f64,
	spd: SpdX,
}

#[derive(PartialOrd, PartialEq)]
#[flux(Sum<f64, 1> = {val} + acc.per(time::SEC))]
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

impl_op!{ *a -> [<PosX as Moment>::Flux; 2] { Pos => a.0 } }

fn pos_speed(pos: &[PosX; 2]) -> f64 {
	let x = pos[0].spd.val;
	let y = pos[1].spd.val;
	(x*x + y*y).sqrt()
}

#[derive(Bundle, Debug)]
struct Dog {
	pos: Pos,
}

#[derive(Component, Debug)]
struct Gun;

const LEFT: i64 = -200;
const RIGHT: i64 = 200;
const TOP: i64 = 200;
const BOTTOM: i64 = -200;
const RADIUS: i64 = 3;

fn setup(world: &mut World) {
    world.spawn(Camera2dBundle::default());
	for mut window in world.query::<&mut Window>().iter_mut(world) {
		// My monitor has a refresh rate of 60hz, so Fifo limits FPS to 60.
		window.present_mode = PresentMode::Immediate;
	}
	// world.resource_mut::<bevy::winit::WinitSettings>().focused_mode = bevy::winit::UpdateMode::Reactive {
	// 	wait: Duration::from_secs_f64(1./60.)
	// };
	
	 // Chime Systems:
	world_add_chime_system(world, when_func_a, do_func_a, temp_default_outlier);
	world_add_chime_system(world, when_func_b, do_func_b, outlier_func_b);
	world_add_chime_system(world, when_func_c, do_func_c, outlier_func_c);
	
	// add_two_dogs(world);
	add_many_dogs(world);
}

#[allow(dead_code)]
fn add_two_dogs(world: &mut World) {
	world.spawn((
		Dog {
			pos: Pos([
				PosX { val: 0., spd: SpdX { val: 000., acc: AccX { val: 0. } } },
				PosX { val: 0., spd: SpdX { val: 20.,  acc: AccX { val: -1000. } } }
			].to_flux(Duration::ZERO)),
		},
		// SpriteBundle {
		// 	transform: Transform::from_xyz(100., 5., 20.),
		// 	texture: world.resource::<AssetServer>().load("textures/air_unit.png"),
		// 	..default()
		// },
		Gun,
	));
	world.spawn((
		Dog {
			pos: Pos([
				PosX { val: 0., spd: SpdX { val: -00., acc: AccX { val: 00. } } },
				PosX { val: 0., spd: SpdX { val: -60., acc: AccX { val: -1000. } } }
			].to_flux(Duration::ZERO)),
		},
		SpriteBundle::default(),
	));
	world.spawn((
		Dog {
			pos: Pos([
				PosX { val: 0., spd: SpdX { val: -00., acc: AccX { val: 00. } } },
				PosX { val: 0., spd: SpdX { val: -40., acc: AccX { val: -1000. } } }
			].to_flux(Duration::ZERO)),
		},
		SpriteBundle::default(),
	));
	// world.spawn((
	// 	Dog {
	// 		pos: Pos([
	// 			PosX { val: 0., spd: SpdX { val: -00., acc: AccX { val: 00. } } },
	// 			PosX { val: 0., spd: SpdX { val: -00., acc: AccX { val: -1000. } } }
	// 		].to_flux(Duration::ZERO)),
	// 	},
	// 	SpriteBundle::default(),
	// ));
	// world.spawn((
	// 	Dog {
	// 		pos: Pos([
	// 			PosX { val: 0., spd: SpdX { val: -00., acc: AccX { val: 00. } } },
	// 			PosX { val: 0., spd: SpdX { val: 20., acc: AccX { val: -1000. } } }
	// 		].to_flux(Duration::ZERO)),
	// 	},
	// 	SpriteBundle::default(),
	// ));
}

#[allow(dead_code)]
fn add_many_dogs(world: &mut World) {
	for x in ((LEFT + RADIUS)..RIGHT).step_by(32) {
	for y in ((BOTTOM + RADIUS)..TOP).step_by(32) {
		world.spawn(
			Dog {
				pos: Pos([
					PosX {
						val: x as f64,
						spd: SpdX {
							val: ((16 + (x.abs() % 32)) * x.signum()) as f64,
							acc: AccX { val: 0. }
						}
					},
					PosX {
						val: y as f64,
						spd: SpdX {
							val: ((16 + (y.abs() % 32)) * y.signum()) as f64,
							acc: AccX { val: -100. }
						}
					}
				].to_flux(Duration::ZERO)),
			}
		);
	}}
}

fn when_func_a(In(mut pred): In<PredCollector<Entity>>, query: Query<(&Pos, Entity), Changed<Pos>>) -> PredCollector<Entity> {
	// let a_time = Instant::now();
	for (pos, entity) in &query {
		let times =
			(pos[0].when_eq(&chime::Constant::from((RIGHT - RADIUS) as f64))/* & pos[0].spd.when(Ordering::Greater, &chime::Constant::from(0.))*/) |
			(pos[0].when_eq(&chime::Constant::from((LEFT  + RADIUS) as f64))/* & pos[0].spd.when(Ordering::Less, &chime::Constant::from(0.))*/);
		pred.add(times, entity);
	}
	// println!("  when_func_a: {:?}", Instant::now().duration_since(a_time));
	pred
}

fn do_func_a(In(ent): In<Entity>, time: Res<Time>, mut query: Query<&mut Pos>) {
	let mut pos = query.get_mut(ent).unwrap();
	let mut pos_x = pos[0].at_mut(time.elapsed());
	pos_x.spd.val *= -0.98;
}

fn when_func_b(In(mut pred): In<PredCollector<Entity>>, query: Query<(&Pos, Entity), Changed<Pos>>/*, time: Res<Time>*/) -> PredCollector<Entity> {
	// let a_time = Instant::now();
	// let time = time.elapsed();
	for (pos, entity) in &query {
		let/* mut*/ times =
			(pos[1].when_eq(&chime::Constant::from((TOP    - RADIUS) as f64)) /*& pos[1].spd.when(Ordering::Greater, &chime::Constant::from(0.))*/) |
			(pos[1].when_eq(&chime::Constant::from((BOTTOM + RADIUS) as f64)) /*& pos[1].spd.when(Ordering::Less, &chime::Constant::from(0.))*/);
		pred.add(times/*.clone()*/, entity);
		// if times.find(|t| *t > time).is_none() && pos[1].at(time).spd.acc.val != 0. {
		// 	println!("Wow! {time:?}, {:?}, {:?}\n  {:?}, spd: {:?}",
		// 		(pos[1].poly(time) - chime::Constant::from((BOTTOM + RADIUS) as f64).into()),
		// 		(pos[1].poly(time) - chime::Constant::from((BOTTOM + RADIUS) as f64).into()).real_roots().collect::<Vec<_>>(),
		// 		pos[1].when_eq(&chime::Constant::from((BOTTOM + RADIUS) as f64)).collect::<Vec<_>>(),
		// 		pos[1].spd.when(Ordering::Less, &chime::Constant::from(0.)).collect::<Vec<_>>()
		// 	);
		// }
	}
	// println!("  when_func_b: {:?}", Instant::now().duration_since(a_time));
	pred
}

fn do_func_b(In(ent): In<Entity>, time: Res<Time>, mut query: Query<&mut Pos>) {
	let mut pos = query.get_mut(ent).unwrap();
	let mut pos_y = pos[1].at_mut(time.elapsed());
	pos_y.spd.val *= -0.98;
	if pos_y.spd.val.abs() < 0.000001 { // > -(2. * pos_y.spd.acc.val.abs()).sqrt()
		pos_y.spd.val = 0.;
		pos_y.spd.acc.val = 0.;
		drop(pos_y);
		pos[0].at_mut(time.elapsed()).spd.val = 0.;
	}
}

fn outlier_func_b(In(ent): In<Entity>, time: Res<Time>, mut query: Query<&mut Pos>) {
	let mut pos = query.get_mut(ent).unwrap();
	let mut pos_y = pos[1].at_mut(time.elapsed());
	pos_y.spd.val = 0.;
	pos_y.spd.acc.val = 0.;
	drop(pos_y);
	pos[0].at_mut(time.elapsed()).spd.val = 0.;
}

fn when_func_c(
	In(mut pred): In<PredCollector<[Entity; 2]>>,
	query: Query<(&Pos, Entity), Changed<Pos>>,
	b_query: Query<(&Pos, Entity)>
) -> PredCollector<[Entity; 2]> {
	// let mut n = 0;
	// let a_time = Instant::now();
	let dis = chime::Constant::from((2 * RADIUS) as f64).poly(Duration::ZERO);
	for (pos, entity) in &query {
		let pos_poly_vec = pos.polys(pos.base_time());
		let time = pos.base_time();
		for (b_pos, b_entity) in &b_query {
			// !!! This kind of thing could be optimized by organizing entities
			// into grid zones, and only making predictions with entities in
			// adjacent zones. Use a prediction case for updating the zones.
			
			// let a_time = Instant::now();
			let b_pos_vec = b_pos.polys(time);
			// println!("A: {:?}", Instant::now().duration_since(a_time));
			// let a_time = Instant::now();
			let times = pos_poly_vec.when_dis_eq(&b_pos_vec, &dis);
			// println!("B: {:?}", Instant::now().duration_since(a_time));
			pred.add(times, [entity/*.min(b_entity)*/, b_entity/*.max(entity)*/]);
		}
		// n += 1;
	}
	// println!("  when_func_c ({n}): {:?}", Instant::now().duration_since(a_time));
	pred
}

fn do_func_c(In(ents): In<[Entity; 2]>, time: Res<Time>, mut query: Query<&mut Pos>) {
	let [mut pos, b_pos] = query.get_many_mut(ents).unwrap();
	let mut pos = pos.at_mut(time.elapsed());
	let b_pos = b_pos.at(time.elapsed());
	let x = pos[0].val - b_pos[0].val;
	let y = pos[1].val - b_pos[1].val;
	let dir = y.atan2(x);
	let spd = pos_speed(&pos) * 0.5;
	pos[0].spd.val = spd * dir.cos();
	pos[1].spd.val = spd * dir.sin();
}

fn outlier_func_c(In(ents): In<[Entity; 2]>, time: Res<Time>, mut query: Query<&mut Pos>) {
	let [mut pos, _b_pos] = query.get_many_mut(ents).unwrap();
	let mut pos = pos.at_mut(time.elapsed());
	pos[0].spd.val = 0.; pos[0].spd.acc.val = 0.;
	pos[1].spd.val = 0.; pos[1].spd.acc.val = 0.;
}

#[allow(dead_code)]
fn discrete_update(mut query: Query<&mut Pos>, time: Res<Time>) {
	let a_time = Instant::now();
	let delta = time.delta().as_secs_f64();
	for mut pos in &mut query {
		*pos[0].spd.val += *pos[0].spd.acc.val * delta / 2.;
		*pos[0].val += *pos[0].spd.val * delta;
		*pos[0].spd.val += *pos[0].spd.acc.val * delta / 2.;
		*pos[1].spd.val += *pos[1].spd.acc.val * delta / 2.;
		*pos[1].val += *pos[1].spd.val * delta;
		*pos[1].spd.val += *pos[1].spd.acc.val * delta / 2.;
		if *pos[0].val >= (RIGHT - RADIUS) as f64 || *pos[0].val <= (LEFT + RADIUS) as f64 {
			*pos[0].val -= 2. * *pos[0].spd.val * delta;
			*pos[0].spd.val *= -1.;
		}
		if *pos[1].val >= (TOP - RADIUS) as f64 || *pos[1].val <= (BOTTOM + RADIUS) as f64 {
			*pos[1].val -= 2. * *pos[1].spd.val * delta;
			*pos[1].spd.val *= -1.;
		}
	}
	let mut n = 0;
	let mut combinations = query.iter_combinations_mut();
	while let Some([mut a, mut b]) = combinations.fetch_next() {
		let x = *a[0].val - *b[0].val;
		let y = *a[1].val - *b[1].val;
		if x*x + y*y <= (4*RADIUS*RADIUS) as f64 {
			let dir = y.atan2(x);
			let h = *a[0].spd.val;
			let v = *a[1].spd.val;
			let spd = (h*h + v*v).sqrt();
			*a[0].spd.val = spd * dir.cos();
			*a[1].spd.val = spd * dir.sin();
			let h = *b[0].spd.val;
			let v = *b[1].spd.val;
			let spd = (h*h + v*v).sqrt();
			*b[0].spd.val = -spd * dir.cos();
			*b[1].spd.val = -spd * dir.sin();
			n += 1;
		}
	}
	println!("discrete at {:?} ({:?}): {:?}", time.elapsed(), n, Instant::now().duration_since(a_time));
}

fn debug_draw(mut draw: Gizmos, time: Res<Time>, pred_time: Res<Time<Chime>>, query: Query<(&Pos, Has<Gun>)>) {
	let s = time.elapsed_seconds();
	for (pos, has_gun) in &query {
		let x = pos[0].at(pred_time.elapsed());
		let y = pos[1].at(pred_time.elapsed());
		let pos = Vec2::new(x.val as f32, y.val as f32);
		draw.circle_2d(pos, RADIUS as f32, Color::YELLOW);
		if has_gun {
			draw.line_2d(
				pos,
				pos + Vec2::new((RADIUS as f32) * s.cos(), (RADIUS as f32) * s.sin()),
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
	draw.rect_2d(
		Vec2::new(((LEFT + RIGHT) / 2) as f32, ((TOP + BOTTOM) / 2) as f32),
		0.0,
		Vec2::new((RIGHT - LEFT) as f32, (TOP - BOTTOM) as f32),
		Color::LIME_GREEN
	);
}

fn main() {
	App::new()
		.add_plugins(DefaultPlugins)
		.add_plugins(ChimePlugin)
		.add_systems(Startup, setup)
		.add_systems(Update, debug_draw)
		// .add_systems(Update, discrete_update)
        // .add_plugins(bevy::diagnostic::LogDiagnosticsPlugin::default())
        // .add_plugins(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default())
		.run();
}

// !!! Make time_try_from_secs always round down
// !!! Make Times and TimeRanges combine the roots in their raw form