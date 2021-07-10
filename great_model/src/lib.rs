use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::{ChiSquared, Distribution, Gamma, StandardNormal};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

#[pyclass]
struct ScaledTModel {
    data_size: usize,
    nu: f64,
    tau2: f64,
    mu: f64,
    alpha2: f64,

    extended_vars: Vec<f64>,
    data: Vec<f64>,
    result_mu: Vec<f64>,
    result_sigma2: Vec<f64>,
}

#[pymethods]
impl ScaledTModel {
    #[new]
    fn new(data: Vec<f64>, nu: f64) -> Self {
        println!("making a rust model");
        ScaledTModel {
            data_size: data.len(),
            nu,
            tau2: 1.0,
            mu: data.iter().sum::<f64>() / (data.len() as f64),
            alpha2: 1.0,
            extended_vars: vec![1.0; data.len()],
            data: data.clone(),
            result_mu: Vec::new(),
            result_sigma2: Vec::new(),
        }
    }

    #[getter]
    fn get_mu(&self) -> PyResult<Vec<f64>> {
        Ok(self.result_mu.clone())
    }

    #[getter]
    fn get_sigma2(&self) -> PyResult<Vec<f64>> {
        Ok(self.result_sigma2.clone())
    }

    fn run(&mut self, burn_in: usize, sample_size: usize) -> PyResult<()> {
        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        for _ in 0..burn_in {
            self.update_extended_vars(&mut rng);
            self.update_alpha2(&mut rng);
            self.update_mu(&mut rng);
            self.update_tau2(&mut rng);
        }

        self.result_mu.resize(sample_size, 0.0);
        self.result_sigma2.resize(sample_size, 0.0);
        for i in 0..sample_size {
            self.update_extended_vars(&mut rng);
            self.update_alpha2(&mut rng);
            self.update_mu(&mut rng);
            self.update_tau2(&mut rng);
            self.result_mu[i] = self.mu;
            self.result_sigma2[i] = self.alpha2 * self.tau2;
        }
        Ok(())
    }
}

fn sample_scaled_inv_chi_square<R>(rng: &mut R, ni: f64, scale: f64) -> f64
where
    R: Rng,
{
    let chi = ChiSquared::new(ni).expect("Failed at creation of ChiSquared");
    ni * scale / chi.sample(rng)
}

impl ScaledTModel {
    fn update_mu<R>(&mut self, rng: &mut R)
    where
        R: Rng,
    {
        let mut tmp: f64;
        let mut variance = 0.0;
        let mut expected_value = 0.0;
        for (datum, ext_var) in self.data.iter().zip(self.extended_vars.iter()) {
            tmp = 1.0 / ext_var;
            variance += tmp;
            expected_value += datum * tmp;
        }
        variance /= self.alpha2;
        expected_value /= self.alpha2;
        variance = 1.0 / variance;
        expected_value = expected_value * variance;

        self.mu = expected_value + variance.sqrt() * rng.sample::<f64, _>(StandardNormal);
    }

    fn update_tau2<R>(&mut self, rng: &mut R)
    where
        R: Rng,
    {
        let x: f64 = self.extended_vars.iter().map(|x| 1.0 / x).sum();
        let scale = 2.0 / (self.nu * x);
        let gamma = Gamma::new((self.data_size as f64) * self.nu / 2.0, scale).expect(&format!(
            "Failed at creation of Gamma, shape:{:}, scale: {:}",
            (self.data_size as f64) * self.nu / 2.0,
            2.0 / (self.nu * x)
        ));
        self.tau2 = gamma.sample(rng);
    }

    fn update_alpha2<R>(&mut self, rng: &mut R)
    where
        R: Rng,
    {
        let mu = self.mu;
        let mut x = self
            .data
            .iter()
            .zip(self.extended_vars.iter())
            .map(|(datum, ext_var)| (datum - mu) * (datum - mu) / ext_var)
            .sum();
        x /= self.data_size as f64;
        self.alpha2 = sample_scaled_inv_chi_square(rng, self.data_size as _, x);
    }

    fn update_extended_vars<R>(&mut self, rng: &mut R)
    where
        R: Rng,
    {
        let mut x: f64;
        let mu = self.mu;
        let alpha2 = self.alpha2;
        let tau2 = self.tau2;
        let nu = self.nu + 1.0;

        let nutau2 = self.nu * tau2;
        let chi = ChiSquared::new(nu).expect("Failed at creation of ChiSquared");
        // let mut res = vec![0.0; self.data_size];
        for i in 0..self.data_size {
            let datum = self.data[i];
            x = (datum - mu) * (datum - mu) * alpha2;
            self.extended_vars[i] = (nutau2 + x) / chi.sample(rng);
        }
        // self.extended_vars.copy_from_slice(&res);
    }
}

#[pymodule]
fn rust_great_model(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ScaledTModel>()?;

    Ok(())
}
