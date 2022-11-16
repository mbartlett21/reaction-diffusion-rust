use core::fmt;

use chrono::Utc;
use minifb::clamp;
use scoped_threadpool::Pool;

#[derive(Default, Debug, Copy, Clone)]
pub struct Cell {
    pub a: f64,
    pub b: f64,
}

#[derive(Debug)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Cell>,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![Cell { a: 1., b: 0. }; width * height],
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

pub struct Simulation {
    pub params: SimulationParams,

    cur_grid: Grid,
    nex_grid: Grid,
    pub framebuffer: Box<[u32]>,
    pool: Pool,
}

impl fmt::Debug for Simulation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Simulation")
            .field("params", &self.params)
            .field("cur_grid", &self.cur_grid)
            .field("nex_grid", &self.nex_grid)
            .field("framebuffer", &self.framebuffer)
            .finish()
    }
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
        framebuffer: vec![0u32; width * height].into_boxed_slice(),
        pool: Pool::new(16),
    }}

    pub fn seed(&mut self) {
        let mut rng = oorandom::Rand64::new(Utc::now().timestamp() as u128);
        let width = self.cur_grid.width;
        let height = self.cur_grid.height;

        let lo = (width * 4) as u64;
        let hi = ((width * height) as u64) - lo;

        // random set of '+' seeds
        let cells = &mut self.cur_grid.cells[..];
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
        let thread_count = self.pool.thread_count() as usize;
        let cell_count = self.cur_grid.width * self.cur_grid.height;

        let cells_per_thread = cell_count / thread_count;

        self.pool.scoped(|s| {
            let width = self.cur_grid.width;
            let height = self.cur_grid.height;

            // How far the current chunk is offset
            let mut offset = 0;

            // Shared between threads
            let cur = &self.cur_grid.cells[..];

            // Split between threads
            let mut nex = &mut self.nex_grid.cells[..];
            let mut frb = &mut self.framebuffer[..];

            while !nex.is_empty() {
                // Split off the chunk for this thread
                let (cnex, rnex) = nex.split_at_mut(cells_per_thread.min(nex.len()));
                let (cfrb, rfrb) = frb.split_at_mut(cells_per_thread.min(frb.len()));
                nex = rnex;
                frb = rfrb;

                let params = self.params;
                s.execute(move || gen_job(width, height, offset, cur, cnex, cfrb, params));
                offset += cells_per_thread;
            }
        });

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
    offset: usize, // How much the offset is from the start
    cur: &[Cell], // Curr isn't offset
    nex: &mut [Cell],
    framebuffer: &mut [u32],
    params: SimulationParams,
) {
    debug_assert!(cur.len() == width * height);
    debug_assert!(cur.len() >= framebuffer.len());
    debug_assert!(nex.len() == framebuffer.len());
    for i in 0..nex.len() {
        let cx = (i + offset) % width;
        let cy = (i + offset) / width;

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