use std::ops::{Deref, DerefMut};
use bevy_chime::*;

use bevy::prelude::*;
use bevy::window::PresentMode;

use chime::*;
use chime::kind::{WhenDisEq, /*WhenDis, WhenEq, When*/};

use std::time::{Duration, /*Instant*/};

#[derive(PartialOrd, PartialEq)]
#[flux(
	kind = "sum::Sum<f64, 2>",
	value = val,
	change = |c| c + spd.per(time::SEC)
)]
#[derive(Component, Clone, Debug)]
struct PosX {
	val: f64,
	spd: SpdX,
}

#[derive(PartialOrd, PartialEq)]
#[flux(
	kind = "sum::Sum<f64, 1>",
	value = val,
	change = |c| c + acc.per(time::SEC)
)]
#[derive(Component, Clone, Debug)]
struct SpdX {
	val: f64,
	acc: AccX,
}

#[derive(PartialOrd, PartialEq)]
#[flux(
	kind = "sum::Sum<f64, 0>",
	value = val,
)]
#[derive(Component, Clone, Debug)]
struct AccX {
	val: f64,
}

#[derive(Component, Clone, Debug)]
struct Pos {
	pos: [<PosX as Moment>::Flux; 2],
	radius: i64,
}

impl Deref for Pos {
	type Target = [<PosX as Moment>::Flux; 2];
	fn deref(&self) -> &Self::Target {
		&self.pos
	}
}

impl DerefMut for Pos {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.pos
	}
}

// fn pos_speed(pos: &[PosX; 2]) -> f64 {
// 	let x = pos[0].spd.val;
// 	let y = pos[1].spd.val;
// 	(x*x + y*y).sqrt()
// }

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

fn setup(world: &mut World) {
    world.spawn(Camera2dBundle::default());
	for mut window in world.query::<&mut Window>().iter_mut(world) {
		// My monitor has a refresh rate of 60hz, so Fifo limits FPS to 60.
		window.present_mode = PresentMode::Immediate;
	}
	// world.resource_mut::<bevy::winit::WinitSettings>().focused_mode = bevy::winit::UpdateMode::Reactive {
	// 	wait: Duration::from_secs_f64(1./60.)
	// };
	
	/* ??? macro syntax:
		#[chime_system]
		fn friction_freeze(query: ChimeQuery<Pos2D>) {
			for pos in query {
				for i in 0..2 {
					when pos.spd.when_index_eq(i, &0.) {
						pos.spd.rate.fric.val[i] = 0;
					} else {
						pos.spd.rate.fric.val[i] = pos.spd.rate.fric.full_val[i];
					}
				}
				// Repetition outlier handler?
			}
		}
	*/
	
	add_two_dogs(world);
	// add_many_dogs(world);
}

#[allow(dead_code)]
fn add_two_dogs(world: &mut World) {
	world.spawn(( // 0v2
		Dog {
			pos: Pos {
				pos: [
					PosX { val: 0., spd: SpdX { val: 00., acc: AccX { val: 0. } } },
					PosX { val: 0., spd: SpdX { val: 20.,  acc: AccX { val: -1000. } } }
				].to_flux_vec(Duration::ZERO),
				radius: 6,
			},
		},
		// SpriteBundle {
		// 	transform: Transform::from_xyz(100., 5., 20.),
		// 	texture: world.resource::<AssetServer>().load("textures/air_unit.png"),
		// 	..default()
		// },
		Gun,
	));
	world.spawn(( // 0v3
		Dog {
			pos: Pos {
				pos: [
					PosX { val: 40., spd: SpdX { val: 100., acc: AccX { val: 0. } } },
					PosX { val: 0., spd: SpdX { val: 20.,  acc: AccX { val: -1000. } } }
				].to_flux_vec(Duration::ZERO),
				radius: 12,
			},
		},
		Gun,
	));
	world.spawn(( // 0v4
		Dog {
			pos: Pos {
				pos: [
					PosX { val: 0.0, spd: SpdX { val: -00., acc: AccX { val: 00. } } },
					PosX { val: -40., spd: SpdX { val: -60., acc: AccX { val: -1000. } } }
				].to_flux_vec(Duration::ZERO),
				radius: 6,
			},
		},
		SpriteBundle::default(),
	));
	world.spawn(( // 0v5
		Dog {
			pos: Pos {
				pos: [
					PosX { val: 0.0, spd: SpdX { val: -00., acc: AccX { val: 00. } } },
					PosX { val: 40., spd: SpdX { val: -40., acc: AccX { val: -1000. } } }
				].to_flux_vec(Duration::ZERO),
				radius: 6,
			},
		},
		SpriteBundle::default(),
	));
}

#[allow(dead_code)]
fn add_many_dogs(world: &mut World) {
	let radius = 3;
	for x in ((LEFT + radius + 1)..RIGHT).step_by(32) {
	for y in ((BOTTOM + radius + 1)..TOP).step_by(32) {
		world.spawn(
			Dog {
				pos: Pos {
					pos: [
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
								acc: AccX { val: 0. }
							}
						}
					].to_flux_vec(Duration::ZERO),
					radius,
				},
			}
		);
	}}
}

fn when_func_a<'w, 's>(mut pred: PredState<'w, 's, Query<'static, 'static, (Ref<'static, Pos>, Entity)>>)
	-> PredState<'w, 's, Query<'static, 'static, (Ref<'static, Pos>, Entity)>>
{
	// let a_time = Instant::now();
	for (case, pos) in pred.iter_mut() {
		let times =
			(pos[0].when_eq(&((RIGHT - pos.radius) as f64))/* & pos[0].spd.when(Ordering::Greater, &0.)*/) |
			(pos[0].when_eq(&((LEFT  + pos.radius) as f64))/* & pos[0].spd.when(Ordering::Less, &0.)*/);
		case.set(times.pre());
	}
	// println!("  when_func_a: {:?}", Instant::now().duration_since(a_time));
	pred
}

fn do_func_a(In(ent): In<Entity>, time: Res<Time>, mut query: Query<&mut Pos>) {
	let mut pos = query.get_mut(ent).unwrap();
	let mut pos_x = pos[0].at_mut(time.elapsed());
	pos_x.spd.val *= -1.;
}

fn when_func_b<'w, 's>(mut pred: PredState<'w, 's, Query<'static, 'static, (Ref<'static, Pos>, Entity)>>)
	-> PredState<'w, 's, Query<'static, 'static, (Ref<'static, Pos>, Entity)>>
{
	// let a_time = Instant::now();
	// let time = time.elapsed();
	for (case, pos) in pred.iter_mut() {
		let/* mut*/ times =
			(pos[1].when_eq(&((TOP    - pos.radius) as f64)) /*& pos[1].spd.when(Ordering::Greater, &0.)*/) |
			(pos[1].when_eq(&((BOTTOM + pos.radius) as f64)) /*& pos[1].spd.when(Ordering::Less, &0.)*/);
		case.set(times.pre()/*.clone()*/);
		// if times.find(|t| *t > time).is_none() && pos[1].at(time).spd.acc.val != 0. {
		// 	println!("Wow! {time:?}, {:?}, {:?}\n  {:?}, spd: {:?}",
		// 		(pos[1].poly(time) - chime::Constant::from((BOTTOM + pos.radius) as f64).into()),
		// 		(pos[1].poly(time) - chime::Constant::from((BOTTOM + pos.radius) as f64).into()).real_roots().collect::<Vec<_>>(),
		// 		pos[1].when_eq(&chime::Constant::from((BOTTOM + pos.radius) as f64)).collect::<Vec<_>>(),
		// 		pos[1].spd.when(Ordering::Less, &chime::Constant::from(0.)).collect::<Vec<_>>()
		// 	);
		// }
	}
	// println!("  when_func_b: {:?}", Instant::now().duration_since(a_time));
	pred
}

fn do_func_b(In(ent): In<Entity>, time: Res<Time>, mut query: Query<&mut Pos>) {
	let mut pos = query.get_mut(ent).unwrap();
	let mut poss = pos.at_vec_mut(time.elapsed());
	let pos_y = &mut poss[1];
	pos_y.spd.val *= -1.;
	if pos_y.spd.val >= 0. && pos_y.spd.val < 1. { // > -(2. * pos_y.spd.acc.val.abs()).sqrt()
		pos_y.spd.val = 0.;
		pos_y.spd.acc.val = 0.;
		poss[0].spd.val = 0.;
		// pos_y.spd.val = 1. * pos_y.spd.val.signum();
	}
	// drop(poss);
	// println!("ground {:?}", (ent, time.elapsed(), 
	// 	pos[0].poly(pos[0].base_time()),
	// 	pos[1].poly(pos[1].base_time()),
	// 	(
	// 		pos[1].when_eq(&chime::Constant::from((TOP    - pos.radius) as f64)) |
	// 		pos[1].when_eq(&chime::Constant::from((BOTTOM + pos.radius) as f64))
	// 	).collect::<Vec<_>>(),
	// ));
}

fn outlier_func_b(In(ent): In<Entity>, time: Res<Time>, mut query: Query<&mut Pos>) {
	let mut pos = query.get_mut(ent).unwrap();
	let mut pos_y = pos[1].at_mut(time.elapsed());
	pos_y.spd.val = 0.;
	pos_y.spd.acc.val = 0.;
	drop(pos_y);
	pos[0].at_mut(time.elapsed()).spd.val = 0.;
	// println!("ground freeze {:?}", (ent, time.elapsed(), 
	// 	pos[0].poly(pos[0].base_time()),
	// 	pos[1].poly(pos[1].base_time())));
}

fn when_func_c<'w, 's>(
	mut pred: PredState<'w, 's, (Query<'static, 'static, (Ref<'static, Pos>, Entity)>, Query<'static, 'static, (Ref<'static, Pos>, Entity)>)>
) -> PredState<'w, 's, (Query<'static, 'static, (Ref<'static, Pos>, Entity)>, Query<'static, 'static, (Ref<'static, Pos>, Entity)>)> {
	// let mut n = 0;
	// let a_time = Instant::now();
	for (case, (pos, b_pos)) in pred.iter_mut() {
		let time = pos.max_base_time();
		let pos_poly_vec = pos.poly_vec(time);
			// !!! This kind of thing could be optimized by organizing entities
			// into grid zones, and only making predictions with entities in
			// adjacent zones. Use a prediction case for updating the zones.
			
			let radius = (pos.radius + b_pos.radius) as f64;
			let dis = chime::Constant::from(radius)
				.poly(Duration::ZERO);
			
			// let a_time = Instant::now();
			let b_pos_vec = b_pos.poly_vec(b_pos.max_base_time());
			// println!("A: {:?}", Instant::now().duration_since(a_time));
			// let a_time = Instant::now();
			let /*mut*/ times = pos_poly_vec.when_dis_eq(b_pos_vec, dis);
			// println!("B: {:?}", Instant::now().duration_since(a_time));
			case.set(times.pre()/*.clone()*/);
			// print!(" -- {:?}", ((entity, b_entity), times.clone().collect::<Vec<_>>()));
			
			// println!("    k0 HERE {:?}", (entity, b_entity, timm.elapsed()));
			// let me = times.clone().collect::<Vec<_>>();
			// println!("    k {:?}", (entity, b_entity, me));
			
			// let poss = pos;
			// let b_poss = b_pos;
			// for (mut t, z) in times.clone() {
			// 	if z >= timm.elapsed() && t.max(timm.elapsed()) - timm.elapsed() <= 20*time::SEC {
			// 		// https://www.desmos.com/calculator/pzgzy75bch
			// 		// https://play.rust-lang.org/?version=nightly&mode=debug&edition=2021&gist=f8d1aa69f2cfca047d6411e0c23ab05d
			// 		t = t.max(timm.elapsed());
			// 		let pos = poss.at(t - time::NANOSEC);
			// 		let b_pos = b_poss.at(t - time::NANOSEC);
			// 		let xx = pos[0].val - b_pos[0].val;
			// 		let yy = pos[1].val - b_pos[1].val;
			// 		let prev = (xx*xx + yy*yy).sqrt();
			// 		drop(pos);
			// 		drop(b_pos);
			// 		let pos = poss.at(t + time::NANOSEC);
			// 		let b_pos = b_poss.at(t + time::NANOSEC);
			// 		let xxx = pos[0].val - b_pos[0].val;
			// 		let yyy = pos[1].val - b_pos[1].val;
			// 		let next = (xxx*xxx + yyy*yyy).sqrt();
			// 		drop(pos);
			// 		drop(b_pos);
			// 		let pos = poss.at(t);
			// 		let b_pos = b_poss.at(t);
			// 		let x = pos[0].val - b_pos[0].val;
			// 		let y = pos[1].val - b_pos[1].val;
			// 		let curr = (x*x + y*y).sqrt();
			// 		drop(pos);
			// 		drop(b_pos);
			// 		let pos = poss.at(t - 2*time::NANOSEC);
			// 		let b_pos = b_poss.at(t - 2*time::NANOSEC);
			// 		let xxx = pos[0].val - b_pos[0].val;
			// 		let yyy = pos[1].val - b_pos[1].val;
			// 		let prev_prev = (xxx*xxx + yyy*yyy).sqrt();
			// 		drop(pos);
			// 		drop(b_pos);
			// 		if curr <= radius /*&& next <= curr*/ /* && prev < 12. && prev_prev >= 12.*/ /*&& next < 12. && next < prev*/ {
			// 			let tt = poss[0].base_time();
			// 			println!("e {:?}", [entity, b_entity]);
			// 			println!("x1: {:?}, y1: {:?}", poss[0].poly(poss[0].base_time()).to_time(tt), poss[1].poly(poss[1].base_time()).to_time(tt));
			// 			println!("x2: {:?}, y2: {:?}", poss[0].poly(t), poss[1].poly(t));
			// 			println!("bx1: {:?}, by1: {:?}", b_poss[0].poly(b_poss[0].base_time()).to_time(tt), b_poss[1].poly(b_poss[1].base_time()).to_time(tt));
			// 			println!("bx2: {:?}, by2: {:?}", b_poss[0].poly(t), b_poss[1].poly(t));
			// 			let dis = *((poss[0].poly(poss[0].base_time()).to_time(tt)-b_poss[0].poly(b_poss[0].base_time()).to_time(tt)).sqr() + (poss[1].poly(poss[1].base_time()).to_time(tt)-b_poss[1].poly(b_poss[1].base_time()).to_time(tt)).sqr());
			// 			println!("dis: {:?}", dis + chime::sum::Sum::<f64, 0>::from(-radius*radius);
			// 			use chime::kind::FluxKind;
			// 			println!("A {:?}: {:?} vs {:?}", t - 2*time::NANOSEC, prev_prev, dis.at(chime::linear::Scalar((t-timm.elapsed()).as_secs_f64()-2e-9)).sqrt());
			// 			println!("B {:?}: {:?} vs {:?}", t - time::NANOSEC, prev, dis.at(chime::linear::Scalar((t-timm.elapsed()).as_secs_f64()-1e-9)).sqrt());
			// 			println!("C {:?}: {:?} vs {:?}", t, curr, dis.at(chime::linear::Scalar((t-timm.elapsed()).as_secs_f64())).sqrt());
			// 			println!("D {:?}: {:?} vs {:?}", t + time::NANOSEC, next, dis.at(chime::linear::Scalar((t-timm.elapsed()).as_secs_f64()+1e-9)).sqrt());
			// 		}
			// 		break
			// 	}
			// }
		// n += 1;
	}
	// println!("  when_func_c ({n}): {:?}", Instant::now().duration_since(a_time));
	pred
}

fn do_func_c(In(ents): In<(Entity, Entity)>, time: Res<Time>, mut query: Query<&mut Pos>) {
	let [mut poss, mut b_poss] = query.get_many_mut([ents.0, ents.1]).unwrap();
	
	let poly = (poss[0].poly(poss[0].base_time()) - b_poss[0].poly(b_poss[0].base_time())).sqr()
		+ (poss[1].poly(poss[1].base_time()) - b_poss[1].poly(b_poss[1].base_time())).sqr();
	assert!(poly.rate_at(time.elapsed()) <= 0., "{:?}", poly);
	
	let mut pos = poss.at_vec_mut(time.elapsed());
	let mut b_pos = b_poss.at_vec_mut(time.elapsed());
	let x = pos[0].val - b_pos[0].val;
	let y = pos[1].val - b_pos[1].val;
	let dir_x = x / x.hypot(y);
	let dir_y = y / x.hypot(y);
	// let spd = pos_speed(&pos) * 0.5;
	// let b_spd = pos_speed(&b_pos) * 0.5;
	// pos[0].spd.val = spd * dir_x;
	// pos[1].spd.val = spd * dir_y;
	// b_pos[0].spd.val = b_spd * -dir_x;
	// b_pos[1].spd.val = b_spd * -dir_y;
	
	fn dot(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
		if (x1 == 0. || x2 == 0.) && (y1 == 0. || y2 == 0.) {
			return 0.
		}
		use accurate::dot::OnlineExactDot;
		use accurate::traits::DotAccumulator;
		let mut a = OnlineExactDot::zero();
		if x1 != 0. && x2 != 0. {
			a = a + (x1, x2);
		}
		if y1 != 0. && y2 != 0. {
			a = a + (y1, y2)
		}
		a.dot()
	}
	
	let mut pos_h_spd = dot(pos[0].spd.val, pos[1].spd.val, dir_x, dir_y);
	let pos_v_spd = dot(pos[1].spd.val, -pos[0].spd.val, dir_x, dir_y);
	let mut b_pos_h_spd = dot(b_pos[0].spd.val, b_pos[1].spd.val, dir_x, dir_y);
	let b_pos_v_spd = dot(b_pos[1].spd.val, -b_pos[0].spd.val, dir_x, dir_y);
	
	let temp = pos_h_spd * 1.;
	pos_h_spd = b_pos_h_spd * 1.;
	b_pos_h_spd = temp;
	if pos_h_spd - b_pos_h_spd < 1e-2 {
		pos_h_spd += 1.;
		b_pos_h_spd -= 1.;
	}
	
	pos[0].spd.val = dot(pos_h_spd, -pos_v_spd, dir_x, dir_y);
	pos[1].spd.val = dot(pos_v_spd, pos_h_spd, dir_x, dir_y);
	b_pos[0].spd.val = dot(b_pos_h_spd, -b_pos_v_spd, dir_x, dir_y);
	b_pos[1].spd.val = dot(b_pos_v_spd, b_pos_h_spd, dir_x, dir_y);
	
	// pos[1].spd.acc.val = -1000.;
	// b_pos[1].spd.acc.val = -1000.;
	
	// let x1 = pos[0].val;
	// let y1 = pos[1].val;
	// let x2 = b_pos[0].val;
	// let y2 = b_pos[1].val;
	// drop(pos);
	// drop(b_pos);
	// let poly = (poss[0].poly(poss[0].base_time()) - b_poss[0].poly(b_poss[0].base_time())).sqr()
	// 	+ (poss[1].poly(poss[1].base_time()) - b_poss[1].poly(b_poss[1].base_time())).sqr();
	// println!("  cool {:?}", (
	// 	ents,
	// 	(pos_h_spd, pos_v_spd),
	// 	(b_pos_h_spd, b_pos_v_spd),
	// 	(x1-x2)*(x1-x2) + (y1-y2)*(y1-y2),
	// 	poss[0].poly(poss[0].base_time()),
	// 	poss[1].poly(poss[1].base_time()),
	// 	b_poss[0].poly(b_poss[0].base_time()),
	// 	b_poss[1].poly(b_poss[1].base_time()),
	// 	poly,
	// 	poss.when_dis_eq(&b_poss.0, &chime::Constant::from(12.)).collect::<Vec<_>>(),
	// 	time.elapsed(),
	// ));
	// assert!(poly.rate_at(time.elapsed()) >= 0., "{:?}", poly);
}

fn outlier_func_c(In(ents): In<(Entity, Entity)>, time: Res<Time>, mut query: Query<&mut Pos>) {
	if ents.0 == ents.1 {
		return
	}
	let [mut poss, mut b_poss] = query.get_many_mut([ents.0, ents.1]).unwrap();
	let mut pos = poss.at_vec_mut(time.elapsed());
	let mut b_pos = b_poss.at_vec_mut(time.elapsed());
	pos[0].spd.val = 0.; pos[0].spd.acc.val = 0.;
	pos[1].spd.val = 0.; pos[1].spd.acc.val = 0.;
	b_pos[0].spd.val = 0.; b_pos[0].spd.acc.val = 0.;
	b_pos[1].spd.val = 0.; b_pos[1].spd.acc.val = 0.;
	
	// let x1 = pos[0].val;
	// let y1 = pos[1].val;
	// let x2 = b_pos[0].val;
	// let y2 = b_pos[1].val;
	// drop(pos);
	// drop(b_pos);
	// println!(" ents {:?}", (
	// 	ents,
	// 	(x1-x2)*(x1-x2) + (y1-y2)*(y1-y2),
	// 	poss[0].poly(poss[0].base_time()),
	// 	poss[1].poly(poss[1].base_time()),
	// 	b_poss[0].poly(b_poss[0].base_time()),
	// 	b_poss[1].poly(b_poss[1].base_time()),
	// 	(poss[0].poly(poss[0].base_time()) - b_poss[0].poly(b_poss[0].base_time())).sqr() + (poss[1].poly(poss[1].base_time()) - b_poss[1].poly(b_poss[1].base_time())).sqr(),
	// 	poss.when_dis_eq(&b_poss.0, &chime::Constant::from(12.)).collect::<Vec<_>>(),
	// 	time.elapsed(),
	// ));
}

#[allow(dead_code)]
fn discrete_update(mut query: Query<&mut Pos>, time: Res<Time>) {
	let delta = time.delta().as_secs_f64();
	for mut pos in &mut query {
		pos[0].spd.val += pos[0].spd.acc.val * delta / 2.;
		pos[0].val += pos[0].spd.val * delta;
		pos[0].spd.val += pos[0].spd.acc.val * delta / 2.;
		pos[1].spd.val += pos[1].spd.acc.val * delta / 2.;
		pos[1].val += pos[1].spd.val * delta;
		pos[1].spd.val += pos[1].spd.acc.val * delta / 2.;
		if pos[0].val >= (RIGHT - pos.radius) as f64 || pos[0].val <= (LEFT + pos.radius) as f64 {
			pos[0].val -= 2. * pos[0].spd.val * delta;
			pos[0].spd.val *= -1.;
		}
		if pos[1].val >= (TOP - pos.radius) as f64 || pos[1].val <= (BOTTOM + pos.radius) as f64 {
			pos[1].val -= 2. * pos[1].spd.val * delta;
			pos[1].spd.val *= -1.;
		}
	}
	let mut combinations = query.iter_combinations_mut();
	while let Some([mut a, mut b]) = combinations.fetch_next() {
		let x = a[0].val - b[0].val;
		let y = a[1].val - b[1].val;
		let radius = (a.radius + b.radius) as f64;
		if x*x + y*y <= radius*radius {
			let dir_x = x / x.hypot(y);
			let dir_y = y / x.hypot(y);
			let h = a[0].spd.val;
			let v = a[1].spd.val;
			let spd = (h*h + v*v).sqrt();
			a[0].spd.val = spd * dir_x;
			a[1].spd.val = spd * dir_y;
			let h = b[0].spd.val;
			let v = b[1].spd.val;
			let spd = (h*h + v*v).sqrt();
			b[0].spd.val = -spd * dir_x;
			b[1].spd.val = -spd * dir_y;
		}
	}
}

fn debug_draw(mut draw: Gizmos, time: Res<Time>, pred_time: Res<Time<Chime>>, query: Query<(&Pos, Has<Gun>)>) {
	let s = time.elapsed_seconds();
	for (pos, has_gun) in &query {
		let x = pos[0].at(pred_time.elapsed());
		let y = pos[1].at(pred_time.elapsed());
		let vec = Vec2::new(x.val as f32, y.val as f32);
		draw.circle_2d(vec, pos.radius as f32, Color::YELLOW);
		if has_gun {
			draw.line_2d(
				vec,
				vec + Vec2::new((pos.radius as f32) * s.cos(), (pos.radius as f32) * s.sin()),
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
		.add_chime_events(ChimeEventBuilder::new(when_func_a).on_begin(do_func_a))
		.add_chime_events(ChimeEventBuilder::new(when_func_b).on_begin(do_func_b).on_repeat(outlier_func_b))
		.add_chime_events(ChimeEventBuilder::new(when_func_c).on_begin(do_func_c).on_repeat(outlier_func_c))
		// .add_systems(Update, discrete_update)
        // .add_plugins(bevy::diagnostic::LogDiagnosticsPlugin::default())
        // .add_plugins(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default())
		.run();
}