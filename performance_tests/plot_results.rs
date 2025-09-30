use plotters::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug)]
struct BenchmarkData {
    num_embeddings: usize,
    insertion_time_ms: f64,
    search_time_ms: f64,
    memory_usage_mb: f64,
}

fn read_csv_data(filename: &str) -> Result<Vec<BenchmarkData>, Box<dyn std::error::Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    
    // Skip header
    let mut lines = reader.lines();
    lines.next();
    
    for line in lines {
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        
        if parts.len() >= 5 {
            let num_embeddings = parts[0].parse::<usize>()?;
            let insertion_time_ms = parts[2].parse::<f64>()?;
            let search_time_ms = parts[3].parse::<f64>()?;
            let memory_usage_mb = parts[4].parse::<f64>()?;
            
            data.push(BenchmarkData {
                num_embeddings,
                insertion_time_ms,
                search_time_ms,
                memory_usage_mb,
            });
        }
    }
    
    Ok(data)
}

fn create_insertion_time_plot(data: &[BenchmarkData]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("performance_tests/insertion_time.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("SWARC Insertion Time vs Dataset Size", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(
            0.0..data.last().unwrap().num_embeddings as f64 * 1.1,
            0.0..data.last().unwrap().insertion_time_ms * 1.1,
        )?;
    
    chart.configure_mesh()
        .x_desc("Number of Embeddings")
        .y_desc("Insertion Time (ms)")
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .draw()?;
    
    let points: Vec<(f64, f64)> = data.iter()
        .map(|d| (d.num_embeddings as f64, d.insertion_time_ms))
        .collect();
    
    chart.draw_series(LineSeries::new(
        points.iter().map(|(x, y)| (*x, *y)),
        RGBColor(255, 0, 0).stroke_width(3),
    ))?;
    
    chart.draw_series(PointSeries::of_element(
        points,
        5,
        RGBColor(255, 0, 0),
        &|c, s, st| {
            return EmptyElement::at(c) + Circle::new((0, 0), s, st.filled());
        },
    ))?;
    
    root.present()?;
    println!("Insertion time plot saved to performance_tests/insertion_time.png");
    Ok(())
}

fn create_search_time_plot(data: &[BenchmarkData]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("performance_tests/search_time.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("SWARC Search Time vs Dataset Size", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(
            0.0..data.last().unwrap().num_embeddings as f64 * 1.1,
            0.0..data.last().unwrap().search_time_ms * 1.1,
        )?;
    
    chart.configure_mesh()
        .x_desc("Number of Embeddings")
        .y_desc("Search Time (ms)")
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .draw()?;
    
    let points: Vec<(f64, f64)> = data.iter()
        .map(|d| (d.num_embeddings as f64, d.search_time_ms))
        .collect();
    
    chart.draw_series(LineSeries::new(
        points.iter().map(|(x, y)| (*x, *y)),
        RGBColor(0, 0, 255).stroke_width(3),
    ))?;
    
    chart.draw_series(PointSeries::of_element(
        points,
        5,
        RGBColor(0, 0, 255),
        &|c, s, st| {
            return EmptyElement::at(c) + Circle::new((0, 0), s, st.filled());
        },
    ))?;
    
    root.present()?;
    println!("Search time plot saved to performance_tests/search_time.png");
    Ok(())
}

fn create_memory_usage_plot(data: &[BenchmarkData]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("performance_tests/memory_usage.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("SWARC Memory Usage vs Dataset Size", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(
            0.0..data.last().unwrap().num_embeddings as f64 * 1.1,
            0.0..data.last().unwrap().memory_usage_mb * 1.1,
        )?;
    
    chart.configure_mesh()
        .x_desc("Number of Embeddings")
        .y_desc("Memory Usage (MB)")
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .draw()?;
    
    let points: Vec<(f64, f64)> = data.iter()
        .map(|d| (d.num_embeddings as f64, d.memory_usage_mb))
        .collect();
    
    chart.draw_series(LineSeries::new(
        points.iter().map(|(x, y)| (*x, *y)),
        RGBColor(0, 255, 0).stroke_width(3),
    ))?;
    
    chart.draw_series(PointSeries::of_element(
        points,
        5,
        RGBColor(0, 255, 0),
        &|c, s, st| {
            return EmptyElement::at(c) + Circle::new((0, 0), s, st.filled());
        },
    ))?;
    
    root.present()?;
    println!("Memory usage plot saved to performance_tests/memory_usage.png");
    Ok(())
}

fn create_throughput_plot(data: &[BenchmarkData]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("performance_tests/throughput.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Calculate throughput (insertions per second)
    let throughput_data: Vec<(f64, f64)> = data.iter()
        .map(|d| {
            let throughput = d.num_embeddings as f64 / (d.insertion_time_ms / 1000.0);
            (d.num_embeddings as f64, throughput)
        })
        .collect();
    
    let max_throughput = throughput_data.iter().map(|(_, t)| *t).fold(0.0, f64::max);
    
    let mut chart = ChartBuilder::on(&root)
        .caption("SWARC Insertion Throughput vs Dataset Size", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(
            0.0..data.last().unwrap().num_embeddings as f64 * 1.1,
            0.0..max_throughput * 1.1,
        )?;
    
    chart.configure_mesh()
        .x_desc("Number of Embeddings")
        .y_desc("Throughput (insertions/sec)")
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .draw()?;
    
    chart.draw_series(LineSeries::new(
        throughput_data.iter().map(|(x, y)| (*x, *y)),
        RGBColor(128, 0, 128).stroke_width(3),
    ))?;
    
    chart.draw_series(PointSeries::of_element(
        throughput_data,
        5,
        RGBColor(128, 0, 128),
        &|c, s, st| {
            return EmptyElement::at(c) + Circle::new((0, 0), s, st.filled());
        },
    ))?;
    
    root.present()?;
    println!("Throughput plot saved to performance_tests/throughput.png");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Reading benchmark data...");
    let data = read_csv_data("performance_tests/benchmark_results.csv")?;
    
    if data.is_empty() {
        eprintln!("No data found in benchmark_results.csv");
        return Ok(());
    }
    
    println!("Creating plots...");
    create_insertion_time_plot(&data)?;
    create_search_time_plot(&data)?;
    create_memory_usage_plot(&data)?;
    create_throughput_plot(&data)?;
    
    println!("All plots created successfully!");
    Ok(())
}
