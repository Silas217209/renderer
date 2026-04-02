mod core;

mod vulkan;
pub mod util;

pub type DefaultBackend = vulkan::VulkanBackend;

pub type Instance = <DefaultBackend as core::Backend>::Instance;
pub type Adapter = <DefaultBackend as core::Backend>::Adapter;
pub type Device = <DefaultBackend as core::Backend>::Device;
