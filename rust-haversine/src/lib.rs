use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::f64::consts::PI;

#[pyfunction]
fn haversine(lon1: f64, lat1: f64, lon2: f64, lat2: f64, miles: Option<bool>) -> PyResult<f64> {
    let earth_radius_km = 6371.0;
    let earth_radius_mi = 3956.0;

    let to_radians = |deg: f64| deg * PI / 180.0;

    let lon1 = to_radians(lon1);
    let lat1 = to_radians(lat1);
    let lon2 = to_radians(lon2);
    let lat2 = to_radians(lat2);

    let dlon = lon2 - lon1;
    let dlat = lat2 - lat1;

    let a = (dlat / 2.0).sin().powi(2)
          + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    let radius = if miles.unwrap_or(false) {
        earth_radius_mi
    } else {
        earth_radius_km
    };

    Ok(c * radius)
}

#[pymodule]
fn rhaversine(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(haversine, m)?)?;
    Ok(())
}