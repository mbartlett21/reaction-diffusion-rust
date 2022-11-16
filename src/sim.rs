use std::ops::Range;
use std::time::{Duration, Instant};
use chrono::Utc;
use minifb::clamp;
use threadpool::ThreadPool;
use crate::util::InfCell;

#[derive(Default, Debug, Copy, Clone)]
pub struct Cell {
    pub a: f64,
    pub b: f64,
}

#[derive(Debug)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub cells: InfCell<Vec<Cell>>,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: InfCell::new(vec![Cell { a: 1., b: 0. }; width * height]),
        }
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct SimulationParams {
    pub   da: f64,
    pub   db: f64,
    pub    f: f64,
    pub    k: f64,
    pub  adj: f64,
    pub diag: f64,
}

#[derive(Debug)]
pub struct Simulation {
    pub params: SimulationParams,

    cur_grid: Grid,
    nex_grid: Grid,
    pub framebuffer: InfCell<Box<[u32]>>,
    pool: ThreadPool,
}

impl Simulation {
    pub fn new(da: f64, db: f64, f: f64, k: f64, adj: f64, diag: f64, width: usize, height: usize) -> Self { Self {
        params: SimulationParams {
            da,
            db,
            f,
            k,
            adj,
            diag,
        },

        cur_grid: Grid::new(width, height),
        nex_grid: Grid::new(width, height),
        framebuffer: InfCell::new(vec![0u32; width * height].into_boxed_slice()),
        pool: ThreadPool::new(16),
    }}

    pub fn seed(&mut self) {
        let mut rng = oorandom::Rand64::new(Utc::now().timestamp() as u128);
        let width = self.cur_grid.width;
        let height = self.cur_grid.height;

        let lo = (width * 4) as u64;
        let hi = ((width * height) as u64) - lo;

        // random set of '+' seeds
        let cells = &mut self.cur_grid.cells.get_mut()[..];
        /*for _ in 0..64 {
            let p = rng.rand_range(lo..hi) as usize;
            cells[p-1].b = 1.0;
            cells[p].b = 1.0;
            cells[p+1].b = 1.0;
            cells[p-width].b = 1.0;
            cells[p+width].b = 1.0;
        }*/
        cells[(cells.len() / 2) + (width / 2)].b = 1.0;
    }

    pub fn generation(&mut self) {
        let thread_count = self.pool.max_count();
        let cell_count = self.cur_grid.width * self.cur_grid.height;

        let cells_per_thread = cell_count / thread_count;

        let width = self.cur_grid.width;
        let height = self.cur_grid.height;

        for i in 0..thread_count {
            let cur = self.cur_grid.cells.get();
            let nex = self.nex_grid.cells.get_mut();
            let fb = self.framebuffer.get_mut();
            let range = (i * cells_per_thread)..((i + 1) * cells_per_thread);
            let params = self.params.clone();

            self.pool.execute(move || gen_job(width, height, cur, nex, fb, range, params));
        }

        if cells_per_thread * thread_count < cell_count {
            let cur = self.cur_grid.cells.get();
            let nex = self.nex_grid.cells.get_mut();
            let fb = self.framebuffer.get_mut();
            let range = (cells_per_thread * thread_count)..cell_count;
            let params = self.params.clone();

            self.pool.execute(move || gen_job(width, height, cur, nex, fb, range, params));
        }

        //println!("");
        //println!("Active: {} of {}", self.pool.active_count(), self.pool.max_count());
        self.pool.join();

        self.swap();
    }

    fn swap(&mut self) {
        std::mem::swap(&mut self.cur_grid, &mut self.nex_grid);
    }
}

fn laplacian(cells: [[&Cell; 3]; 3], params: SimulationParams) -> (f64, f64) {
    // cells are indexed [y][x]
    // + is down for y
    let u = cells[0][1]; // 0,-
    let d = cells[2][1]; // 0,+
    let l = cells[1][0]; // -,0
    let r = cells[1][2]; // +,0

    let lu = cells[0][0]; // -,-
    let ru = cells[0][2]; // +,-
    let ld = cells[2][0]; // -,+
    let rd = cells[2][2]; // +,+

    let cell = cells[1][1]; // 0,0

    (
        -cell.a + ((u.a + d.a + l.a + r.a) * params.adj) + ((lu.a + ru.a + ld.a + rd.a) * params.diag),
        -cell.b + ((u.b + d.b + l.b + r.b) * params.adj) + ((lu.b + ru.b + ld.b + rd.b) * params.diag)
    )
}

fn gen_job(
    width: usize,
    height: usize,
    cur: &[Cell],
    nex: &mut [Cell],
    framebuffer: &mut [u32],
    range: Range<usize>,
    params: SimulationParams,
) {
    debug_assert!(cur.len() == nex.len() && cur.len() == framebuffer.len());
    for i in range {
        let cx = i % width;
        let cy = i / width;

        let cells = [
            [
                // (-, -)
                &cur[(cx + width - 1) % width + ((cy + height - 1) % height) * width],
                // (0, -)
                &cur[cx + ((cy + height - 1) % height) * width],
                // (+, -)
                &cur[(cx + 1) % width + ((cy + height - 1) % height) * width],
            ],
            [
                // (-, 0)
                &cur[(cx + width - 1) % width + cy * width],
                // (0, 0)
                &cur[cx + cy * width],
                // (+, 0)
                &cur[(cx + 1) % width + cy * width],
            ],
            [
                // (-, +)
                &cur[(cx + width - 1) % width + ((cy + 1) % height) * width],
                // (0, +)
                &cur[cx + ((cy + 1) % height) * width],
                // (+, +)
                &cur[(cx + 1) % width + ((cy + 1) % height) * width],
            ],
        ];
        let c = cells[1][1];
        let abb = c.a * c.b * c.b;
        let (lapa, lapb) = laplacian(cells, params);

        let n = &mut nex[i];
        n.a = clamp(0.0, c.a + (params.da * lapa) - abb + (params.f * (1.0 - c.a)), 1.0);
        n.b = clamp(0.0, c.b + (params.db * lapb) + abb - ((params.k + params.f) * c.b), 1.0);

        let val = clamp(0.1f64, (n.a + n.b) * (n.a - n.b) * 2f64, 1.0);
        framebuffer[i] = from_f64_rgb(val, val, val);
    }
}

fn from_f64_rgb(r: f64, g: f64, b: f64) -> u32 {
    (((r * 255.0) as u32) << 16) | (((g * 255.0) as u32) << 8) | ((b * 255.0) as u32)
}