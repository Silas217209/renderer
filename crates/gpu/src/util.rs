use raw_window_handle::HasDisplayHandle;

pub struct InstanceDescriptor<'a> {
    pub display_handle: Option<&'a dyn HasDisplayHandle>,
}