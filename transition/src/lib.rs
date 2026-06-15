use rand::distr::{Distribution, Uniform};
// SM: at least 48 KiB = 48 * 1024 B = 49,152 B

// 384 pairs max. 
pub type PRICE = u64;
pub const POSITIONS_SIZE: usize = 1000_000_000;

pub fn generate_market_conditions() -> (u64, Vec<PRICE>){
    let price: u64 = 100_000;
    let mut positions: Vec<PRICE> = Vec::new();
    for _ in 0..POSITIONS_SIZE{
        positions.push(get_random_price());
    }
    (price, positions)
}

fn get_random_price() -> u64{
    let mut rng = rand::rng();
    let dist = Uniform::new_inclusive(1000u64, 1_000_000u64).expect("Failed to get distribution");
    dist.sample(&mut rng)
}

fn get_random_idx(size: usize) -> u32{
    let mut rng = rand::rng();
    let dist = Uniform::new(0, size).expect("Failed to get distribution");
    dist.sample(&mut rng) as u32
}

#[cfg(test)]
mod state_transition_tests{
    use crate::{generate_market_conditions};
    use std::time::Instant;
    #[test]
    fn benchmark_cpu_liquidations(){
        let start = Instant::now();
        let market_conditions = generate_market_conditions();
        // check for each position whether price is lower than or equal to liq
        let price = market_conditions.0;
        let positions = market_conditions.1;
        let mut out: Vec<u8> = Vec::new();
        // loop over all positions and compare to idx price
        for position in positions{
            // this is assuming just long positions
            if price <= position{
                out.push(1);
            }
            else{
                out.push(0);
            }
        }
        let duration = start.elapsed();
        println!("CPU Liquidations took: {} ms", duration.as_millis());
    }
}
