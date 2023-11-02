use faer::prelude::*;
use faer::solvers::Qr;
use faer::{IntoFaer, IntoNdarray};
// use faer::polars::{polars_to_faer_f64, Frame};
use ndarray::{Array2, ArrayView2};
use num;
use polars::prelude::*;
use polars_core::utils::rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[polars_expr(output_type=Int64)]
fn pl_gcd(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].i64()?;
    let ca2 = inputs[1].i64()?;

    if ca2.len() == 1 {
        let b = ca2.get(0).unwrap();
        let out: Int64Chunked = ca1
            .into_iter()
            .map(|op_a| {
                if let Some(a) = op_a {
                    Some(num::integer::gcd(a, b))
                } else {
                    None
                }
            })
            .collect();
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Int64Chunked = ca1
            .into_iter()
            .zip(ca2.into_iter())
            .map(|(op_a, op_b)| {
                if let (Some(a), Some(b)) = (op_a, op_b) {
                    Some(num::integer::gcd(a, b))
                } else {
                    None
                }
            })
            .collect();
        Ok(out.into_series())
    } else {
        Err(PolarsError::ComputeError(
            "Inputs must have the same length.".into(),
        ))
    }
}

#[polars_expr(output_type=Int64)]
fn pl_lcm(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].i64()?;
    let ca2 = inputs[1].i64()?;

    if ca2.len() == 1 {
        let b = ca2.get(0).unwrap();
        let out: Int64Chunked = ca1
            .into_iter()
            .map(|op_a| {
                if let Some(a) = op_a {
                    Some(num::integer::lcm(a, b))
                } else {
                    None
                }
            })
            .collect();
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Int64Chunked = ca1
            .into_iter()
            .zip(ca2.into_iter())
            .map(|(op_a, op_b)| {
                if let (Some(a), Some(b)) = (op_a, op_b) {
                    Some(num::integer::lcm(a, b))
                } else {
                    None
                }
            })
            .collect();
        Ok(out.into_series())
    } else {
        Err(PolarsError::ComputeError(
            "Inputs must have the same length.".into(),
        ))
    }
}

// I am not sure this is right. I still don't quite understand the purpose of this.
fn lstsq_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "betas",
        DataType::Struct(
            input_fields[1..]
                .iter()
                .map(|f| Field::new(&format!("beta_{}", f.name()), DataType::Float64))
                .collect(),
        ),
    ))
}

#[derive(Deserialize)]
struct LstsqKwargs {
    drop_null_regressors: bool,
}

#[polars_expr(output_type_func=lstsq_output)]
fn lstsq(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    // TODO: Use the input series names to name the output series when provided.
    // let beta_names: Vec<String> = (0..(inputs.len()-1)).map(|i| format!("x{i}")).collect();
    //
    let nrows = inputs[0].len();
    let ncols = inputs.len() - 2;

    // y
    let y = inputs[0].f64()?;
    let y = y.to_ndarray()?;

    // W
    let weights = inputs[1].f64()?;
    let weights = weights.to_ndarray()?;
    let mut weights = weights.mapv(|x| x.sqrt());

    // broadcast weights to len of y if pl.literal passed
    if weights.len() == 1 {
        let weight = weights[0];
        weights = Array1::from_elem(nrows, weight);
    }

    // create mask of cols to keep that defaults to all true
    let mut mask = vec![true; ncols];

    if kwargs.drop_null_regressors {
        // Get a boolean mask of which rows have no null values
        mask = inputs[2..]
            .iter()
            .map(|s| s.is_not_null().all())
            .collect::<Vec<bool>>();
    } else {
        // check if there are any columns with a null value and if so, raise an error
        let null_cols = inputs[2..]
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if s.is_null().any() { Some(i) } else { None })
            .collect::<Vec<usize>>();
        if !null_cols.is_empty() {
            return PolarsResult::Err(PolarsError::ComputeError(
                format!(
                    "Columns {:?} have null values.
                Use drop_null_regressors=True to drop these columns during regression
                or handle null values before passing to regression.",
                    null_cols
                )
                .into(),
            ));
        }
    }

    let used_regressor_ind: Vec<_> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &val)| if val { Some(i) } else { None })
        .collect();

    let dropped_regressor_ind: Vec<_> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &val)| if val { None } else { Some(i) })
        .collect();

    // apply mask to inputs to only get the inputs where mask is true
    let x_series = inputs[2..].to_vec();
    let x_series = x_series
        .into_iter()
        .zip(mask)
        .enumerate()
        .filter_map(|(i, (x, m))| {
            if m {
                Some(x.with_name(&format!("x{i}"))) // This is currently required to work with `over` expressions.
            } else {
                None
            }
        })
        .collect::<Vec<Series>>();

    // X
    let df_x = DataFrame::new(x_series)?;

    match df_x.to_ndarray::<Float64Type>(IndexOrder::Fortran) {
        Ok(x) => {
            // Calculating weighted matrices
            let xw = (x.t().to_owned() * &weights).reversed_axes();
            let yw = y.to_owned() * &weights;

            // Convert to faer for the solve
            let xw = xw.view().into_faer();
            let yw = yw.into_shape((nrows, 1)).unwrap();
            let yw = yw.view().into_faer();
            let x = x.view().into_faer();
            let y = y.into_shape((nrows, 1)).unwrap();
            let y = y.view().into_faer();

            // Solving Least Square, without bias term
            let betas = Qr::new(xw).solve_lstsq(yw);
            let preds = x * &betas;
            let preds_array = preds.as_ref().into_ndarray();
            let resid = y - &preds;
            let resid_array: ArrayView2<f64> = resid.as_ref().into_ndarray();
            let betas = betas.as_ref().into_ndarray();

            let mut out_series: Vec<Series> = Vec::with_capacity(betas.len() + 2);

            //Add calculated betas to output
            for (b, i) in betas.into_iter().zip(used_regressor_ind) {
                out_series.push(
                    // A copy
                    Series::from_vec(&format!("x{i}"), vec![*b; nrows]),
                );
            }
            // Add dropped regressors as nulls in place of the beta
            for i in dropped_regressor_ind {
                out_series.push(Series::new_null(&format!("x{i}"), nrows));
            }
            out_series.push(
                // A copy
                Series::from_iter(preds_array).with_name("y_pred"),
            );
            out_series.push(
                // A copy
                Series::from_iter(resid_array).with_name("resid"),
            );
            let out = StructChunked::new("results", &out_series)?.into_series();
            Ok(out)
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=lstsq_output)]
fn pl_lstsq2(inputs: &[Series]) -> PolarsResult<Series> {
    // let beta_names: Vec<String> = (0..(inputs.len()-1)).map(|i| format!("x{i}")).collect();
    let nrows = inputs[0].len();
    // let ncols = inputs.len() - 1;

    // y
    let y = inputs[0].f64()?;
    let y = y.to_ndarray()?.into_shape((nrows, 1)).unwrap();
    let y = y.view().into_faer();

    // X
    let df_x = DataFrame::new(inputs[1..].to_vec())?;

    match df_x.to_ndarray::<Float64Type>(IndexOrder::Fortran) {
        Ok(x) => {
            // Solving Least Square, without bias term
            let x = x.view().into_faer();
            Qr::new(x);
            let betas = Qr::new(x).solve_lstsq(y);
            let preds = x * &betas;
            let preds_array = preds.as_ref().into_ndarray();
            let resid = y - &preds;
            let resid_array: ArrayView2<f64> = resid.as_ref().into_ndarray();
            let betas = betas.as_ref().into_ndarray();

            let mut out_series: Vec<Series> = Vec::with_capacity(betas.len() + 2);
            for (i, b) in betas.into_iter().enumerate() {
                out_series.push(
                    // A copy
                    Series::from_vec(&format!("x{i}"), vec![*b; nrows]),
                );
            }
            out_series.push(
                // A copy
                Series::from_iter(preds_array).with_name("y_pred"),
            );
            out_series.push(
                // A copy
                Series::from_iter(resid_array).with_name("resid"),
            );

            let out = StructChunked::new("results", &out_series)?.into_series();
            Ok(out)
        }
        Err(e) => Err(e),
    }
}

// #[polars_expr(output_type_func=lstsq_output)]
// fn lstsq2(inputs: &[Series]) -> PolarsResult<Series> {
//     // Iterate over the inputs and name each one with .with_name() and collect them into a vector
//     let mut series_vec = Vec::new();

//     // Have to name each one because they don't have names if passed in via .over()
//     for (i, series) in inputs[1..].iter().enumerate() {
//         let series = series.clone().with_name(&format!("x{i}"));
//         series_vec.push(series);
//     }
//     let beta_names: Vec<String> = series_vec.iter().map(|s| s.name().to_string()).collect();

//     let y = &inputs[0];

//     let df_y = df!(y.name() => y)?
//         .lazy();
//     let mat_y = polars_to_faer_f64(df_y);

//     let y = &inputs[0]
//         .f64()
//         .unwrap()
//         .to_ndarray()
//         .unwrap()
//         .to_owned()
//         .into_shape((inputs[0].len(), 1))
//         .unwrap();
//     let y = y.view().into_faer();

//     // Create a polars DataFrame from the input series
//     let todf = DataFrame::new(series_vec);
//     match todf {
//         Ok(df) => {
//             let x = df
//                 .to_ndarray::<Float64Type>(IndexOrder::Fortran)
//                 .unwrap()
//                 .to_owned();
//             let x = x.view().into_faer();
//             Qr::new(x);
//             let betas = Qr::new(x).solve_lstsq(y);
//             let preds = x * &betas;
//             let preds_array = preds.as_ref().into_ndarray();
//             let resids = y - &preds;
//             let resid_array: ArrayView2<f64> = resids.as_ref().into_ndarray();
//             let betas = betas.as_ref().into_ndarray();

//             let mut out_series: Vec<Series> = betas
//                 .iter()
//                 .zip(beta_names.iter())
//                 .map(|(beta, name)| Series::new(name, vec![*beta; inputs[0].len()]))
//                 .collect();
//             // Add a series of residuals and y_pred to the output
//             let y_pred_series =
//                 Series::new("y_pred", preds_array.iter().copied().collect::<Vec<f64>>());
//             let resid_series =
//                 Series::new("resid", resid_array.iter().copied().collect::<Vec<f64>>());
//             out_series.push(y_pred_series);
//             out_series.push(resid_series);
//             let out = StructChunked::new("results", &out_series)?.into_series();
//             Ok(out)
//         }
//         Err(e) => {
//             println!("Error: {}", e);
//             PolarsResult::Err(e)
//         }
//     }
// }
