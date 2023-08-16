include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub fn update_neurons(neurons: &mut [f32], inputs: &[f32]) {
    unsafe {
        execute_update_neurons(neurons.as_mut_ptr(), inputs.as_ptr(), neurons.len() as i32);
    }
}
