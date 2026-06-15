use std::ffi::c_void;

use transition::POSITIONS_SIZE;

pub struct GpuState {
    pub d_prices: *mut c_void,
    pub d_liq: *mut c_void,
    pub d_out: *mut c_void,
    pub n: usize,
}

#[allow(unused)]
impl GpuState {
    unsafe fn new(prices: &[u64], liq: &[u64]) -> Self {
        let mut d_prices: *mut c_void = std::ptr::null_mut();
        let mut d_liq: *mut c_void = std::ptr::null_mut();
        let mut d_out: *mut c_void = std::ptr::null_mut();
        unsafe {
            cudaMalloc(&mut d_prices, prices.len() * 8);
            cudaMalloc(&mut d_liq, liq.len() * 8);
            cudaMalloc(&mut d_out, liq.len()); // u8 output

            cudaMemcpy(d_prices, prices.as_ptr() as *const _, prices.len() * 8, 1);

            cudaMemcpy(d_liq, liq.as_ptr() as *const _, liq.len() * 8, 1);
        }

        Self {
            d_prices,
            d_liq,
            d_out,
            n: liq.len(),
        }
    }
    unsafe fn run(&self, price: u64) {
        let mut out: Vec<u8> = vec![0; POSITIONS_SIZE];
        unsafe {
            launch_check_liquidations(
                self.d_liq as *const u64,
                price as *const u64,
                self.d_out as *mut u8,
                self.n as u32,
            );
            cudaMemcpy(out.as_mut_ptr() as *mut _, self.d_out, out.len(), 2);
        }
    }
}

#[link(name = "cudart")]
unsafe extern "C" {
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> u32;

    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: u32) -> u32;

    pub fn cudaFree(ptr: *mut c_void) -> u32;

    pub fn launch_add_kernel(a: *mut u32, b: *mut u32, out: *mut u32, n: u32);

    pub unsafe fn launch_check_liquidations(
        liq: *const u64,
        prices: *const u64,
        out: *mut u8,
        n: u32,
    );
}

pub const CUDA_MEMCPY_HOST_TO_DEVICE: u32 = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: u32 = 2;

#[cfg(test)]
mod cuda_tests {
    use super::*;
    use std::{
        time::Instant,
    };
    use transition::{POSITIONS_SIZE, generate_market_conditions};
    #[test]
    fn kernel_test() {
        let n = 1024usize;

        let a: Vec<u32> = (0..n).map(|x| x as u32).collect();
        let b: Vec<u32> = (0..n).map(|x| (x * 2) as u32).collect();
        let mut out: Vec<u32> = vec![0; n];

        unsafe {
            let mut d_a: *mut c_void = std::ptr::null_mut();
            let mut d_b: *mut c_void = std::ptr::null_mut();
            let mut d_out: *mut c_void = std::ptr::null_mut();

            let bytes = n * std::mem::size_of::<u32>();

            cudaMalloc(&mut d_a, bytes);
            cudaMalloc(&mut d_b, bytes);
            cudaMalloc(&mut d_out, bytes);

            cudaMemcpy(
                d_a,
                a.as_ptr() as *const c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );

            cudaMemcpy(
                d_b,
                b.as_ptr() as *const c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );

            launch_add_kernel(
                d_a as *mut u32,
                d_b as *mut u32,
                d_out as *mut u32,
                n as u32,
            );

            cudaMemcpy(
                out.as_mut_ptr() as *mut c_void,
                d_out,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );

            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_out);
        }
        println!("out[0] = {}", out[0]);
        println!("out[10] = {}", out[10]);
        println!("out[100] = {}", out[100]);
    }

    #[test]
    fn liquidation_test() {
        let start = Instant::now();
        let market_data = generate_market_conditions();
        let price = market_data.0;
        let liq_flat: Vec<u64> = market_data.1;
        let mut out: Vec<u8> = vec![0; POSITIONS_SIZE];
        let mut d_liq: *mut c_void = std::ptr::null_mut();
        let mut d_price: *mut c_void = std::ptr::null_mut();
        let mut d_out: *mut c_void = std::ptr::null_mut();

        unsafe {
            cudaMalloc(&mut d_liq, liq_flat.len() * 8);
            cudaMalloc(&mut d_price, 8);
            cudaMalloc(&mut d_out, out.len());
            cudaMemcpy(d_liq, liq_flat.as_ptr() as *const _, liq_flat.len() * 8, 1);
            cudaMemcpy(
                d_price,
                &price as *const u64 as *const c_void,
                std::mem::size_of::<u64>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );
            launch_check_liquidations(
                d_liq as *const u64,
                d_price as *const u64,
                d_out as *mut u8,
                POSITIONS_SIZE as u32,
            );
            cudaMemcpy(out.as_mut_ptr() as *mut _, d_out, out.len(), 2);
        }
        let duration = start.elapsed();
        println!("CUDA Liquidations took: {} ms", duration.as_millis());
    }

    #[test]
    fn liquidation_test_with_state() {
        unsafe {
            let state = GpuState::new(&[100_000], &generate_market_conditions().1.to_vec());
            state.run(100_000);

            // now re-use and bench
            let start = Instant::now();
            for _ in 0..1 {
                state.run(100_000);
            }
            println!("CUDA Liquidations took: {} ms", start.elapsed().as_millis());
        }
    }
}
