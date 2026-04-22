use nalgebra::{DMatrix, SymmetricEigen};
use plotters::prelude::*;

const N: usize = 1000; // grid points
const L: f64 = 20.0;  // box size [-L/2, L/2]

/// Different potential well types
pub enum Potential {
    InfiniteSquareWell,
    FiniteSquareWell { width: f64, depth: f64 },
    HarmonicOscillator { k: f64 },
}

impl Potential {
    fn value(&self, x: f64) -> f64 {
        match self {
            Self::InfiniteSquareWell => 0.0, // Walls are implicitly infinite naturally by boundary
            Self::FiniteSquareWell { width, depth } => {
                if x.abs() < *width / 2.0 { 0.0 } else { *depth }
            }
            Self::HarmonicOscillator { k } => 0.5 * k * x * x,
        }
    }
}

/// Solves TISE H psi = E psi
fn solve_tise(potential: &Potential, n_states: usize) -> Vec<(f64, Vec<f64>)> {
    let dx = L / (N as f64 - 1.0);
    let mut h = DMatrix::<f64>::zeros(N, N);
    let t_coeff = 1.0 / (2.0 * dx * dx); // Using hbar=1, m=1
    
    for i in 0..N {
        let x = -L / 2.0 + (i as f64) * dx;
        let v_i = potential.value(x);
        
        h[(i, i)] = 2.0 * t_coeff + v_i;
        if i > 0 {
            h[(i, i - 1)] = -t_coeff;
            h[(i - 1, i)] = -t_coeff;
        }
    }
    
    let eig = SymmetricEigen::new(h);
    let mut evals: Vec<(usize, f64)> = eig.eigenvalues.iter().copied().enumerate().collect();
    evals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    let mut results = Vec::new();
    for i in 0..n_states.min(N) {
        let idx = evals[i].0;
        let energy = evals[i].1;
        
        // Extract eigenvector
        let col = eig.eigenvectors.column(idx);
        let mut psi: Vec<f64> = col.iter().copied().collect();
        
        // Normalize psi: sum(|psi|^2 dx) = 1
        let norm_sq: f64 = psi.iter().map(|&p| p * p * dx).sum();
        let norm = norm_sq.sqrt();
        for p in &mut psi { *p /= norm; }
        
        // Ensure consistent phase (first large peak is positive)
        let max_elem = psi.iter().max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap()).unwrap();
        if *max_elem < 0.0 {
            for p in &mut psi { *p = -*p; }
        }

        results.push((energy, psi));
    }
    
    results
}

fn plot_results(
    potential: &Potential, 
    results: &[(f64, Vec<f64>)], 
    filename: &str, 
    title: &str,
    y_max: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let dx = L / (N as f64 - 1.0);
    let xs: Vec<f64> = (0..N).map(|i| -L / 2.0 + (i as f64) * dx).collect();
    let vs: Vec<f64> = xs.iter().map(|&x| potential.value(x)).collect();
    
    let root = BitMapBackend::new(filename, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let max_x = L / 2.0;
    let min_x = -max_x;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(min_x..max_x, 0.0..y_max)?;

    chart.configure_mesh().draw()?;

    // Plot potential
    chart.draw_series(LineSeries::new(
        xs.iter().zip(vs.iter()).filter(|&(_, &v)| v <= y_max + 10.0).map(|(&x, &v)| (x, v)),
        &BLACK.mix(0.8),
    ))?
    .label("Potential V(x)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &RGBColor(255, 140, 0)];
    let factor = 3.0; // scaling factor for probability density

    // Plot states
    for (i, (energy, psi)) in results.iter().enumerate() {
        let color = colors[i % colors.len()];
        
        let path: Vec<(f64, f64)> = xs.iter().zip(psi.iter())
            .map(|(&x, &p)| (x, *energy + factor * p * p))
            .collect();
            
        chart.draw_series(LineSeries::new(path, color))?
            .label(format!("E_{} = {:.3}", i, energy))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
            
        // Plot energy level line
        chart.draw_series(LineSeries::new(
            vec![(min_x, *energy), (max_x, *energy)],
            color.mix(0.3),
        ))?;
    }

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_states = 6;
    
    println!("Solving Infinite Square Well...");
    let isw = Potential::InfiniteSquareWell;
    let results_isw = solve_tise(&isw, num_states);
    plot_results(&isw, &results_isw, "infinite_square_well.png", "Infinite Square Well", results_isw.last().unwrap().0 + 5.0)?;
    
    println!("Solving Finite Square Well...");
    let fsw = Potential::FiniteSquareWell { width: 5.0, depth: 15.0 };
    let results_fsw = solve_tise(&fsw, num_states);
    plot_results(&fsw, &results_fsw, "finite_square_well.png", "Finite Square Well", fsw.value(10.0).max(results_fsw.last().unwrap().0) + 5.0)?;
    
    println!("Solving Quantum Harmonic Oscillator...");
    let qho = Potential::HarmonicOscillator { k: 1.0 };
    let results_qho = solve_tise(&qho, num_states);
    plot_results(&qho, &results_qho, "harmonic_oscillator.png", "Quantum Harmonic Oscillator", results_qho.last().unwrap().0 + 5.0)?;
    
    println!("Plots generated successfully.");
    Ok(())
}
