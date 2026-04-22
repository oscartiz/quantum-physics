use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex;
use plotters::prelude::*;
use std::f64::consts::PI;

const N: usize = 1000;
const L: f64 = 60.0; // wider box

pub enum Potential {
    InfiniteSquareWell,
    FiniteSquareWell { width: f64, depth: f64 },
    HarmonicOscillator { k: f64 },
    FiniteBarrier { width: f64, height: f64 },
}

impl Potential {
    fn value(&self, x: f64) -> f64 {
        match self {
            Self::InfiniteSquareWell => 0.0,
            Self::FiniteSquareWell { width, depth } => {
                if x.abs() < *width / 2.0 { 0.0 } else { *depth }
            }
            Self::HarmonicOscillator { k } => 0.5 * k * x * x,
            Self::FiniteBarrier { width, height } => {
                if x.abs() < *width / 2.0 { *height } else { 0.0 }
            }
        }
    }
}

fn solve_tise(potential: &Potential, n_states: usize) -> Vec<(f64, Vec<f64>)> {
    let dx = L / (N as f64 - 1.0);
    let mut h = DMatrix::<f64>::zeros(N, N);
    let t_coeff = 1.0 / (2.0 * dx * dx); 
    
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
        let col = eig.eigenvectors.column(idx);
        let mut psi: Vec<f64> = col.iter().copied().collect();
        let norm_sq: f64 = psi.iter().map(|&p| p * p * dx).sum();
        let norm = norm_sq.sqrt();
        for p in &mut psi { *p /= norm; }
        
        let mut max_abs = 0.0;
        let mut max_elem = 0.0;
        for &p in &psi {
            if p.abs() > max_abs {
                max_abs = p.abs();
                max_elem = p;
            }
        }
        
        if max_elem < 0.0 {
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
    
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(-max_x..max_x, 0.0..y_max)?;
    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        xs.iter().zip(vs.iter()).filter(|&(_, &v)| v <= y_max + 10.0).map(|(&x, &v)| (x, v)),
        &BLACK.mix(0.8),
    ))?.label("Potential V(x)").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &RGBColor(255, 140, 0)];
    let factor = 3.0; 

    for (i, (energy, psi)) in results.iter().enumerate() {
        let color = colors[i % colors.len()];
        let path: Vec<(f64, f64)> = xs.iter().zip(psi.iter())
            .map(|(&x, &p)| (x, *energy + factor * p * p)).collect();
        chart.draw_series(LineSeries::new(path, color))?
            .label(format!("E_{} = {:.3}", i, energy))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        chart.draw_series(LineSeries::new(vec![(-max_x, *energy), (max_x, *energy)], color.mix(0.3)))?;
    }
    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).position(SeriesLabelPosition::UpperRight).draw()?;
    root.present()?;
    Ok(())
}

fn simulate_tunneling() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simulating tunneling... Solving complete eigensystem (this takes a few seconds)");
    // A barrier at the center with width 2 and height 12.0
    let barrier = Potential::FiniteBarrier { width: 3.0, height: 12.0 };
    let dx = L / (N as f64 - 1.0);
    
    // We get 600 states to form a good detailed basis for the localized wavepacket
    let num_states = 600;
    let results = solve_tise(&barrier, num_states);
    
    // Initial Wavepacket properties
    let x0 = -15.0;            // start on the left
    let sigma: f64 = 2.0;      // width
    // Average Energy roughly = k0^2 / 2 = 12.5 (a bit over or around the barrier)
    let k0: f64 = 5.0;         // initial momentum
    
    // Construct psi_0
    let mut psi_0 = vec![Complex::new(0.0, 0.0); N];
    let mut norm_sq = 0.0;
    for i in 0..N {
        let x = -L / 2.0 + (i as f64) * dx;
        let envelope = (-0.5 * ((x - x0) / sigma).powi(2)).exp();
        psi_0[i] = Complex::new(envelope * (k0 * x).cos(), envelope * (k0 * x).sin());
        norm_sq += psi_0[i].norm_sqr() * dx;
    }
    let norm = norm_sq.sqrt();
    for p in &mut psi_0 { *p /= norm; }
    
    // c_n = <psi_n | psi_0>
    let mut c_n = vec![Complex::new(0.0, 0.0); num_states];
    for n in 0..num_states {
        let mut c = Complex::new(0.0, 0.0);
        for i in 0..N {
            c += psi_0[i] * results[n].1[i] * dx;
        }
        c_n[n] = c;
    }
    
    // Render loop
    println!("Rendering gif...");
    let root = BitMapBackend::gif("tunneling.gif", (1000, 600), 100)?.into_drawing_area();
    let num_frames = 150;
    let dt = 0.02;
    let max_x = 25.0; // zoom in somewhat
    let min_x = -max_x;
    
    let xs: Vec<f64> = (0..N).map(|i| -L / 2.0 + (i as f64) * dx).collect();
    let vs: Vec<f64> = xs.iter().map(|&x| barrier.value(x)).collect();
    let v_path: Vec<(f64, f64)> = xs.iter().zip(vs.iter()).filter(|&(_, &v)| v <= 15.0).map(|(&x, &v)| (x, v)).collect();
    
    for frame in 0..num_frames {
        let t = (frame as f64) * dt;
        
        let mut psi_t = vec![Complex::new(0.0, 0.0); N];
        for n in 0..num_states {
            let energy = results[n].0;
            // e^{-i E_n t}
            let phase = -energy * t;
            let time_factor = c_n[n] * Complex::new(phase.cos(), phase.sin());
            for i in 0..N {
                psi_t[i] += time_factor * results[n].1[i];
            }
        }
        
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption(format!("Quantum Tunneling (t = {:.2})", t), ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(min_x..max_x, 0.0..18.0)?;
            
        chart.configure_mesh().draw()?;
        
        chart.draw_series(LineSeries::new(v_path.clone(), &BLACK.mix(0.8)))?
            .label("Barrier V(x)");
            
        let density: Vec<(f64, f64)> = xs.iter().zip(psi_t.iter())
            .map(|(&x, &p)| (x, p.norm_sqr() * 25.0)) // scale probability up for visibility
            .collect();
            
        chart.draw_series(AreaSeries::new(density.clone(), 0.0, &BLUE.mix(0.2)))?;
        chart.draw_series(LineSeries::new(density, &BLUE))?
            .label("Probability Density |ψ|²");
            
        chart.configure_series_labels().position(SeriesLabelPosition::UpperRight).draw()?;
        root.present()?;
    }
    
    println!("tunneling.gif generated.");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_states = 5;
    
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
    
    simulate_tunneling()?;
    
    println!("All plots generated successfully.");
    Ok(())
}
