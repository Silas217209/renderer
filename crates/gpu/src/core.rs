pub trait Backend: Sized + Clone {
    type Instance: Instance<Self>;
    type Adapter: Adapter<Self>;
    type Device: Device<Self>;
}
pub trait Instance<B: Backend> {
    fn new(descriptor: &crate::util::InstanceDescriptor) -> Self;
}

pub trait Adapter<B: Backend> {}

pub trait Device<B: Backend> {}
